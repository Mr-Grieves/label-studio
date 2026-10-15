"""Label Studio ML backend for effsam: isonoai's click-prompted SAM.

Runs the Keras SAM logged to MLflow by an isonoai training run (see main.py), split
the way isonoai's interactive mode uses it:
    sam.image_encoder  images (B, 256, 256, 1) -> embeddings
    sam.decode         embeddings + clicks + previous mask -> mask logits and IoU

Interaction: KeyPointLabels clicks in, a BrushLabels mask out. Regular clicks are
foreground, alt-clicks (is_positive=False) are background. Box prompts are
ignored -- the model was trained on clicks only.

Per click, Label Studio re-sends every click made so far. The backend:
  - embeds each image once and caches the embedding, so a click only runs the
    (cheap) decoder;
  - feeds the previous prediction's logits back in as the mask prompt when the
    new clicks extend the previous ones, as in training ("mask loopback"), and
    predicts without it otherwise (first click, or a click deleted/moved).
Both caches are in-process, so run this backend with a single gunicorn worker.

Preprocessing matches isonoai's training pipeline: grayscale, /255, bilinear
resize to 256x256. Clicks are given to the model in pixels of that 256 frame.
"""

import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

import cv2
import keras
import mlflow
import numpy as np
import tensorflow as tf
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from pyconvert.lib import pyconvert

from tsam.utils.pyconvert_utils import (
    convert_mask_to_postscan,
    convert_point_to_prescan,
    decode_images_json,
    get_prescan_size,
)
from tsam.utils.s3_utils import get_s3_client

# Importing it registers the SAM classes with Keras, so the saved model can be rebuilt
from tsam.utils.sam import VIT_LAYERS
from tsam.utils.urls import extract_s3_key_from_url

logger = logging.getLogger(__name__)

ALPHA = 127
LS_WIDTH, LS_HEIGHT = 800, 800
INPUT_SIZE = 256  # the model's input frame (as trained); clicks are in its pixels
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", 16))
LOGITS_CACHE_SIZE = int(os.getenv("LOGITS_CACHE_SIZE", 256))
SMOOTH_SIGMA = float(os.getenv("SMOOTH_SIGMA", 2.5)) # Gaussian smoothing of the logits, in pixels of the 256 frame; 0 = off


class LRUCache:
    """A small thread-safe least-recently-used cache."""

    def __init__(self, size: int):
        self.size = size
        self.items = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.items:
                return None
            self.items.move_to_end(key)
            return self.items[key]

    def put(self, key, value):
        with self.lock:
            self.items[key] = value
            self.items.move_to_end(key)
            while len(self.items) > self.size:
                self.items.popitem(last=False)


def preprocess_xy(x_percent: float, y_percent: float, timestamp: int, convert_utils: pyconvert.ConvertUtils) -> tuple:
    """Convert the post-scan clicks to prescan image coordinates."""
    # Convert to prescan
    x = x_percent / 100 * LS_WIDTH
    y = y_percent / 100 * LS_HEIGHT
    x, y = convert_point_to_prescan(np.array([x, y]), timestamp, LS_WIDTH, LS_HEIGHT, convert_utils)
    return x, y


def largest_component(mask: np.ndarray) -> np.ndarray:
    """The largest 8-connected component of a binary mask, as a bool mask (all False if empty)."""
    n, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    return components == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def refine_logits(logits: np.ndarray) -> np.ndarray:
    """Smooth the (256, 256) logits and keep only the largest connected foreground component.

    Works on the logits rather than a binary mask, so the result can still be looped
    back as the mask prompt -- the model was trained on its own logits, not 0/1 masks.
    """
    if SMOOTH_SIGMA > 0:
        logits = cv2.GaussianBlur(logits, (0, 0), SMOOTH_SIGMA)
    # Dropped components become background with the same confidence.
    return np.where(largest_component(logits > 0), logits, -np.abs(logits))


def postprocess_mask(
    logits: np.ndarray, ts: int, pre_w: int, pre_h: int, convert_utils: pyconvert.ConvertUtils
) -> np.ndarray:
    """Convert (256, 256) logits to a binary mask in the post-scan (Label Studio) frame."""
    # Upsample the logits, not a binary mask (as SAM does), and hand the scan conversion
    # a soft mask, so the edge is placed after interpolation rather than staircased.
    # cv2 takes (width, height): the result is (pre_h, pre_w), the prescan image's (lines, samples per line).
    prescan_logits = cv2.resize(logits, (pre_w, pre_h), interpolation=cv2.INTER_LINEAR)
    prescan_prob = 1.0 / (1.0 + np.exp(-prescan_logits))
    postscan_mask = convert_mask_to_postscan(prescan_prob, ts, LS_WIDTH, LS_HEIGHT, convert_utils)
    # Scan conversion can shrink thin parts of the mask below a pixel (e.g. a linear
    # probe's samples are squeezed ~2x into the depth axis), breaking off small
    # islands; keep the main piece, as in refine_logits().
    return largest_component(postscan_mask).astype(np.uint8)


def load_sam(mlflow_id: str) -> keras.Model:
    """Load the SAM logged by isonoai training run `mlflow_id`.

    Downloads the run's model artifact and loads its .keras file with Keras directly, rather than with
    mlflow.tensorflow.load_model: alongside the model, MLflow pickled Keras's registered custom objects, and
    unpickling them imports isonoai. Keras itself finds those classes by their registered names, which
    tsam.utils.sam provides.
    """
    logger.info(f"Loading SAM from MLflow run {mlflow_id}")
    artifact_dir = Path(mlflow.artifacts.download_artifacts(f"runs:/{mlflow_id}/model"))
    return keras.saving.load_model(artifact_dir / "data" / "model.keras", custom_objects=VIT_LAYERS, compile=False)


class EffSamHandler:
    """Everything that outlives a request: the models, the caches and the pre-scan converter.

    Thread-safe: requests run on several threads at once.
    """

    def __init__(self, mlflow_id, embedding_cache_size: int, logits_cache_size: int):
        self.sam = load_sam(mlflow_id)
        self.embeddings = LRUCache(embedding_cache_size)  # prescan S3 key -> embeddings
        self.logits = LRUCache(logits_cache_size)  # (task id, image url, clicks) -> that prediction's logits
        # (blob S3 key, timestamp) -> (images JSON string, prescan (w, h))
        self.metadata = LRUCache(embedding_cache_size)

        # One converter. It holds one image's metadata at a time (set by
        # loadImageMetadata), so loading it and converting that image's clicks
        # happen together under converter_lock.
        self.convert_utils = pyconvert.ConvertUtils()
        self.converter_lock = threading.Lock()
        self.converter_key = None  # (blob S3 key, timestamp) whose metadata convert_utils holds

        self.s3_client = get_s3_client()

    def preprocess_img(self, prescan_key: str) -> tf.Tensor:
        """Prescan image at an S3 key -> (1, 256, 256, 1) float32 in [0, 1]."""
        # Decoded exactly as training decodes pre_scan_image: one channel, then /255.
        image = self.s3_client.get_image(prescan_key)
        pixels = tf.cast(image, tf.float32)[None] / 255.0
        return tf.image.resize(pixels, (INPUT_SIZE, INPUT_SIZE), method="bilinear")

    def embed(self, prescan_key: str) -> tf.Tensor:
        """The prescan image's embeddings, cached under its S3 key."""
        cached = self.embeddings.get(prescan_key)
        if cached is None:
            cached = self.sam.image_encoder(self.preprocess_img(prescan_key), training=False)
            self.embeddings.put(prescan_key, cached)
        return cached

    def load_metadata(self, blob_key: str, timestamp: int) -> tuple:
        """The image's metadata from its blob JSON, cached: (key, images JSON string, prescan (w, h))."""
        key = (blob_key, timestamp)
        cached = self.metadata.get(key)
        if cached is None:
            image_blob = self.s3_client.get_json_blob(blob_key)
            images_json = decode_images_json(image_blob, timestamp=timestamp)
            cached = (json.dumps(images_json), get_prescan_size(images_json))
            self.metadata.put(key, cached)
        return (key, *cached)

    def convert_clicks_to_prescan(self, clicks_percent: list, blob_key: str, timestamp: int) -> list:
        """Label Studio's post-scan clicks (percent) -> (x, y) pixels of the model's 256 frame,
        via the image's pre-scan coordinates."""
        key, images_json, (pre_w, pre_h) = self.load_metadata(blob_key, timestamp)
        with self.converter_lock:
            if self.converter_key != key:
                self.convert_utils.loadImageMetadata(images_json)
                self.converter_key = key
            points = []
            for x_percent, y_percent in clicks_percent:
                x, y = preprocess_xy(x_percent, y_percent, timestamp, self.convert_utils)
                # Normalize to [0, INPUT_SIZE] based on the prescan image size
                points.append((x / pre_w * INPUT_SIZE, y / pre_h * INPUT_SIZE))
        return points

    def convert_mask_to_postscan(self, logits: np.ndarray, blob_key: str, timestamp: int) -> np.ndarray:
        """(256, 256) logits from the model's frame -> binary mask in the post-scan frame."""
        key, images_json, (pre_w, pre_h) = self.load_metadata(blob_key, timestamp)
        with self.converter_lock:
            if self.converter_key != key:
                self.convert_utils.loadImageMetadata(images_json)
                self.converter_key = key
            postscan_mask = postprocess_mask(logits, timestamp, pre_w, pre_h, self.convert_utils)
        return postscan_mask


class NewModel(LabelStudioMLBase):
    """effsam: isonoai's SAM, prompted with clicks.

    label-studio-ml creates a new instance per request, so the handler that should
    outlive a request is a class attribute shared by all instances.
    """

    # Set once at startup by main.py (`NewModel.configure`), before the server takes requests.
    handler: EffSamHandler | None = None
    mlflow_id: str | None = None

    @classmethod
    def configure(cls, mlflow_id: str) -> None:
        """Load the model from MLflow run `mlflow_id`; every instance then shares it."""
        cls.handler = EffSamHandler(mlflow_id, EMBEDDING_CACHE_SIZE, LOGITS_CACHE_SIZE)
        cls.mlflow_id = mlflow_id

    def setup(self):
        """Report the MLflow run as the model version, so predictions say which model made them."""
        self.set("model_version", f"effsam-{self.mlflow_id}")

    def predict(self, tasks: list[dict], context: dict | None = None, **kwargs) -> ModelResponse:
        """Returns the mask for the clicks made so far on this image."""
        from_name, to_name, value = self.get_first_tag_occurence("BrushLabels", "Image")
        if not context or not context.get("result"):
            logger.warning("No context! -- cannot make a prediction.")
            return ModelResponse(predictions=[])  # no interaction yet

        kps = [ctx for ctx in context["result"] if ctx["type"] == "keypointlabels"]
        if not kps:
            logger.info("effsam only uses click prompts -- Box inputs not supported.")
            return ModelResponse(predictions=[])

        # The task holds an s3:// URL to {prefix}/postscan-image_{inst}_{exam}_{image uuid}_{timestamp}.jpg; the
        # prescan image and the image's blob JSON sit beside it.
        task = tasks[0]
        image_url = task["data"][value]
        ts = int(task["data"]["pre_scan_center_b_timestamp"])
        postscan_key = extract_s3_key_from_url(image_url)
        prefix, filename = postscan_key.rsplit("/", 1)
        *_, image_uuid, filename_ts = Path(filename).stem.split("_")
        assert int(filename_ts) == ts, f"Task timestamp {ts} doesn't match its image {filename}"
        prescan_key = f"{prefix}/{filename.replace('postscan-image', 'prescan-image', 1)}"
        blob_key = f"{prefix}/image-blob_{image_uuid}.json"

        # Compute embeddings for the image using the handler.
        embeddings = self.handler.embed(prescan_key)

        # Clicks in Label Studio's percent coordinates -> pixels of the model's 256 frame.
        pts = self.handler.convert_clicks_to_prescan(
            [(ctx["value"]["x"], ctx["value"]["y"]) for ctx in kps], blob_key, ts
        )
        clicks = [(x, y, 1.0 if ctx.get("is_positive", True) else 0.0) for (x, y), ctx in zip(pts, kps)]
        label = kps[-1]["value"]["keypointlabels"][0]

        # Mask loopback: reuse the previous prediction if these clicks extend its clicks by one.
        clicks = tuple(clicks)
        previous = self.handler.logits.get((task.get("id"), image_url, clicks[:-1])) if len(clicks) > 1 else None
        raw_logits, iou = self.handler.sam.decode(
            embeddings,
            tf.constant([[c[:2] for c in clicks]], dtype=tf.float32),  # (1, N, 2)
            tf.constant([[c[2] for c in clicks]], dtype=tf.float32),  # (1, N)
            previous,  # (1, 4E, 4E) logits, or None for no mask prompt
        )
        # Smooth and keep the largest component; these refined logits are both shown and looped back.
        logits = refine_logits(np.asarray(raw_logits)[0].astype(np.float32))
        self.handler.logits.put((task.get("id"), image_url, clicks), tf.constant(logits[None]))

        # Low-res logits -> mask in Label Studio's post-scan frame.
        postscan_mask = self.handler.convert_mask_to_postscan(logits, blob_key, ts)
        score = float(np.clip(np.asarray(iou)[0], 0.0, 1.0))  # the IoU head's estimate of this mask's quality

        return ModelResponse(
            predictions=[
                {
                    "result": [
                        {
                            "id": str(uuid4())[:4],
                            "from_name": from_name,
                            "to_name": to_name,
                            # The RLE is of the post-scan frame; export reshapes it by these dimensions.
                            "original_width": LS_WIDTH,
                            "original_height": LS_HEIGHT,
                            "image_rotation": 0,
                            "value": {
                                "format": "rle",
                                "rle": brush.mask2rle(postscan_mask * ALPHA),
                                "brushlabels": [label],
                            },
                            "score": score,
                            "type": "brushlabels",
                            "readonly": False,
                        }
                    ],
                    "model_version": self.get("model_version"),
                    "score": score,
                }
            ]
        )
