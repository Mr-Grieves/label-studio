"""Label Studio ML backend for effsam: isonoai's click-prompted SAM.

Runs the two TF SavedModels written by export_savedmodel.py, so it needs only
TensorFlow -- no isonoai code, no MLflow:
    MODEL_DIR/encoder  images (B, 256, 256, 1) -> embeddings
    MODEL_DIR/decoder  embeddings + clicks + previous mask -> mask logits and IoU

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
from typing import Dict, List, Optional
from uuid import uuid4

import cv2
import numpy as np
import tensorflow as tf
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from PIL import Image
from pyconvert.lib import pyconvert
from pyconvert_utils import (
    convert_mask_to_postscan,
    convert_point_to_prescan,
    decode_images_json,
    get_prescan_size,
)

logger = logging.getLogger(__name__)

ALPHA = 127
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
LS_WIDTH, LS_HEIGHT = 800, 800
INPUT_SIZE = 256  # the model's input frame (as trained); clicks are in its pixels
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", 16))
LOGITS_CACHE_SIZE = int(os.getenv("LOGITS_CACHE_SIZE", 256))
SMOOTH_SIGMA = float(os.getenv("SMOOTH_SIGMA", 1.0))  # Gaussian smoothing of the logits, in pixels of the 256 frame; 0 = off


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


def preprocess_img(path: str):
    """Image file -> (1, 256, 256, 1) float32 in [0, 1], plus its original (H, W)."""
    # Retrieve the prescan image for this task
    prescan_path = path.replace("postscan", "prescan")
    image = Image.open(prescan_path).convert("L")
    width, height = image.size
    pixels = np.asarray(image, dtype=np.float32)[None, ..., None] / 255.0
    return tf.image.resize(pixels, (INPUT_SIZE, INPUT_SIZE), method="bilinear"), height, width


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


def postprocess_mask(logits: np.ndarray, ts: int, pre_w: int, pre_h: int, convert_utils: pyconvert.ConvertUtils) -> np.ndarray:
    """Convert (256, 256) logits to a binary mask in the post-scan (Label Studio) frame."""
    # Upsample the logits, not a binary mask (as SAM does), and hand the scan conversion
    # a soft mask, so the edge is placed after interpolation rather than staircased.
    prescan_logits = cv2.resize(logits, (pre_h, pre_w), interpolation=cv2.INTER_LINEAR)
    prescan_prob = 1.0 / (1.0 + np.exp(-prescan_logits))
    postscan_mask = convert_mask_to_postscan(prescan_prob, ts, LS_WIDTH, LS_HEIGHT, convert_utils)
    # Scan conversion can shrink thin parts of the mask below a pixel (e.g. a linear
    # probe's samples are squeezed ~2x into the depth axis), breaking off small
    # islands; keep the main piece, as in refine_logits().
    return largest_component(postscan_mask).astype(np.uint8)


class EffSamHandler:
    """Everything that outlives a request: the models, the caches and the pre-scan converter.

    Thread-safe: requests run on several threads at once.
    """

    def __init__(self, model_dir: str, embedding_cache_size: int, logits_cache_size: int):
        self.encoder = tf.saved_model.load(os.path.join(model_dir, "encoder"))
        self.decoder = tf.saved_model.load(os.path.join(model_dir, "decoder"))
        self.embeddings = LRUCache(embedding_cache_size)  # (task id, image url) -> (embeddings, H, W)
        self.logits = LRUCache(logits_cache_size)          # (task id, image url, clicks) -> that prediction's logits
        self.metadata = LRUCache(embedding_cache_size)     # (blob path, timestamp) -> (images JSON string, prescan (w, h))

        # One converter. It holds one image's metadata at a time (set by
        # loadImageMetadata), so loading it and converting that image's clicks
        # happen together under converter_lock.
        self.convert_utils = pyconvert.ConvertUtils()
        self.converter_lock = threading.Lock()
        self.converter_key = None  # (blob path, timestamp) whose metadata convert_utils holds

    def embed(self, key: tuple, image_path: Path) -> tuple:
        """The image's embeddings and original (H, W), cached under `key`."""
        cached = self.embeddings.get(key)
        if cached is None:
            images, height, width = preprocess_img(str(image_path))
            cached = (self.encoder.serve([images]), height, width)
            self.embeddings.put(key, cached)
        return cached

    def load_metadata(self, blob_path: Path, timestamp: int) -> tuple:
        """The image's metadata from its blob JSON, cached: (key, images JSON string, prescan (w, h))."""
        key = (str(blob_path), timestamp)
        cached = self.metadata.get(key)
        if cached is None:
            with open(blob_path, "r") as f:
                image_blob = json.load(f)
            images_json = decode_images_json(image_blob, timestamp=timestamp)
            cached = (json.dumps(images_json), get_prescan_size(images_json))
            self.metadata.put(key, cached)
        return (key, *cached)

    def convert_clicks_to_prescan(self, clicks_percent: list, blob_path: Path, timestamp: int) -> list:
        """Label Studio's post-scan clicks (percent) -> (x, y) pixels of the model's 256 frame,
        via the image's pre-scan coordinates."""
        key, images_json, (pre_w, pre_h) = self.load_metadata(blob_path, timestamp)
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

    def convert_mask_to_postscan(self, logits: np.ndarray, blob_path: Path, timestamp: int) -> np.ndarray:
        """(256, 256) logits from the model's frame -> binary mask in the post-scan frame."""
        key, images_json, (pre_w, pre_h) = self.load_metadata(blob_path, timestamp)
        with self.converter_lock:
            if self.converter_key != key:
                self.convert_utils.loadImageMetadata(images_json)
                self.converter_key = key
            postscan_mask = postprocess_mask(logits, timestamp, pre_w, pre_h, self.convert_utils)
        return postscan_mask


class NewModel(LabelStudioMLBase):
    """effsam: isonoai's SAM, prompted with clicks.

    label-studio-ml creates a new instance per request, so the handler that should
    outlive a request lives in one EffSamState shared by all instances.
    """

    # `handler` is a class-attribute. It is only created once and is shared across all instances of NewModel.
    handler = EffSamHandler(MODEL_DIR, EMBEDDING_CACHE_SIZE, LOGITS_CACHE_SIZE)

    def predict(self, tasks: list[dict], context: dict | None = None, **kwargs) -> ModelResponse:
        """Returns the mask for the clicks made so far on this image."""
        from_name, to_name, value = self.get_first_tag_occurence("BrushLabels", "Image")
        if not context or not context.get("result"):
            logger.warning("No context! -- cannot make a prediction.")
            return ModelResponse(predictions=[])  # no interaction yet

        keypoints = [ctx for ctx in context["result"] if ctx["type"] == "keypointlabels"]
        if not keypoints:
            logger.info("effsam only uses click prompts -- Box inputs not supported.")
            return ModelResponse(predictions=[])

        task = tasks[0]
        image_url = task["data"][value]
        # The task holds a URL (/data/local-files/?d=...); the image's blob JSON sits next to the file it resolves to.
        image_path = Path(self.get_local_path(image_url, task_id=task.get("id")))
        filename_parts = image_path.stem.split("_")
        image_uuid, timestamp = filename_parts[3], int(filename_parts[4])
        blob_path = image_path.parent / f"image-blob_{image_uuid}.json"
        embeddings, height, width = self.handler.embed((task.get("id"), image_url), image_path)

        # Clicks in Label Studio's percent coordinates -> pixels of the model's 256 frame.
        points = self.handler.convert_clicks_to_prescan(
            [(ctx["value"]["x"], ctx["value"]["y"]) for ctx in keypoints], blob_path, timestamp
        )
        clicks = [
            (x, y, 1.0 if ctx.get("is_positive", True) else 0.0) for (x, y), ctx in zip(points, keypoints)
        ]
        label = keypoints[-1]["value"]["keypointlabels"][0]

        # Mask loopback: reuse the previous prediction if these clicks extend its clicks by one.
        clicks = tuple(clicks)
        previous = self.handler.logits.get((task.get("id"), image_url, clicks[:-1])) if len(clicks) > 1 else None
        out = self.handler.decoder.serve({
            "embeddings": embeddings,
            "points": tf.constant([[c[:2] for c in clicks]], dtype=tf.float32),   # (1, N, 2)
            "labels": tf.constant([[c[2] for c in clicks]], dtype=tf.float32),    # (1, N)
            "pred_mask": previous if previous is not None else tf.zeros((1, 4 * embeddings.shape[1], 4 * embeddings.shape[2])),
            "has_mask": tf.constant([0.0 if previous is None else 1.0]),
        })
        # Smooth and keep the largest component; these refined logits are both shown and looped back.
        logits = refine_logits(out["logits"].numpy()[0].astype(np.float32))
        self.handler.logits.put((task.get("id"), image_url, clicks), tf.constant(logits[None]))

        # Low-res logits -> mask in Label Studio's post-scan frame.
        postscan_mask = self.handler.convert_mask_to_postscan(logits, blob_path, timestamp)
        score = float(np.clip(out["iou"].numpy()[0], 0.0, 1.0))  # the IoU head's estimate of this mask's quality

        return ModelResponse(predictions=[{
            "result": [{
                "id": str(uuid4())[:4],
                "from_name": from_name,
                "to_name": to_name,
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {"format": "rle", "rle": brush.mask2rle(postscan_mask * ALPHA), "brushlabels": [label]},
                "score": score,
                "type": "brushlabels",
                "readonly": False,
            }],
            "model_version": self.get("model_version"),
            "score": score,
        }])
