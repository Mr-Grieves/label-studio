"""Label Studio ML backend for TSAM: isonoai's box-prompted SAM.

Runs the box-only Keras SAM logged to MLflow by an isonoai training run (see main.py), in two steps:
    sam.image_encoder                    prescan image (B, 256, 256, 1) -> embeddings
    sam.prompt_encoder, sam.mask_decoder  embeddings + box -> mask logits and IoU

Interaction: a smart RectangleLabels box in, a BrushLabels mask out (with the box's label, so the two
tags need the same label values). Each box gives one prediction: the model takes no clicks and no
previous mask. Keypoint clicks are ignored.

The backend embeds each image once and caches the embedding, so a new box only runs the (cheap)
prompt encoder and mask decoder. The cache is in-process, so run this backend with a single
gunicorn worker.

Online training (online.py, online-training.md): Label Studio's webhook on a submitted annotation
calls fit(), which turns its polygons into training pairs and queues them; predictions use the
trainer's live decoder, with the box's structure. A new run first backfills: it feeds the project's
latest annotations through fit() at startup, as if they had just been submitted.

Preprocessing matches isonoai's training pipeline: grayscale, /255, bilinear resize to 256x256.
The box is given to the model as (x0, y0, x1, y1) in pixels of that 256 frame of the prescan image,
converted from its two post-scan corners -- which assumes a linear probe, where a post-scan
rectangle is also one in prescan. (On a curvilinear probe its sides would curve.)
"""

import json
import logging
import os
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path
from uuid import uuid4

import cv2
import keras
import mlflow
import numpy as np
import requests
import tensorflow as tf
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from pyconvert.lib import pyconvert

from tsam.config import Config
from tsam.data import INPUT_SIZE, label_name
from tsam.online import OnlineTrainer
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
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", 16))
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


def box_corners(value: dict) -> np.ndarray:
    """A Label Studio rectangle's top-left and bottom-right corners, as (2, 2) pixels of the post-scan frame.

    Label Studio gives x, y, width and height in percent. Rotation is ignored: on a linear probe the
    post-scan -> prescan conversion is a scale and shift, so the two corners fix the whole box.
    """
    x, y = value["x"] / 100 * LS_WIDTH, value["y"] / 100 * LS_HEIGHT
    w, h = value["width"] / 100 * LS_WIDTH, value["height"] / 100 * LS_HEIGHT
    return np.array([[x, y], [x + w, y + h]])


def largest_component(mask: np.ndarray) -> np.ndarray:
    """The largest 8-connected component of a binary mask, as a bool mask (all False if empty)."""
    n, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    return components == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def refine_logits(logits: np.ndarray) -> np.ndarray:
    """Smooth the (256, 256) logits and keep only the largest connected foreground component."""
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
    """Load the SAM logged by MLflow run `mlflow_id`: an isonoai training run, or an online-training run (online.py).

    Downloads the run's model artifact and loads its .keras file with Keras directly, rather than with
    mlflow.tensorflow.load_model: alongside the model, MLflow pickled Keras's registered custom objects, and
    unpickling them imports isonoai. Keras itself finds those classes by their registered names, which
    tsam.utils.sam provides.
    """
    logger.info(f"Loading SAM from MLflow run {mlflow_id}")
    artifact_dir = Path(mlflow.artifacts.download_artifacts(f"runs:/{mlflow_id}/model"))
    return keras.saving.load_model(artifact_dir / "data" / "model.keras", custom_objects=VIT_LAYERS, compile=False)


class TSAMHandler:
    """Everything that outlives a request: the models, the caches and the pre-scan converter.

    Thread-safe: requests run on several threads at once.
    """

    def __init__(self, mlflow_id, embedding_cache_size: int):
        self.sam = load_sam(mlflow_id)
        self.embeddings = LRUCache(embedding_cache_size)  # prescan S3 key -> embeddings
        # (blob S3 key, timestamp) -> (images JSON string, prescan (w, h))
        self.metadata = LRUCache(embedding_cache_size)

        # One converter. It holds one image's metadata at a time (set by
        # loadImageMetadata), so loading it and converting that image's box or mask
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

    def locate(self, task_data: dict, image_url: str) -> tuple[str, str, int]:
        """A task's prescan image key, blob JSON key and timestamp.

        The task holds an s3:// URL to {prefix}/postscan-image_{inst}_{exam}_{image uuid}_{timestamp}.jpg; the
        prescan image and the image's blob JSON sit beside it.
        """
        ts = int(task_data["pre_scan_center_b_timestamp"])
        prefix, filename = extract_s3_key_from_url(image_url).rsplit("/", 1)
        *_, image_uuid, filename_ts = Path(filename).stem.split("_")
        assert int(filename_ts) == ts, f"Task timestamp {ts} doesn't match its image {filename}"
        prescan_key = f"{prefix}/{filename.replace('postscan-image', 'prescan-image', 1)}"
        return prescan_key, f"{prefix}/image-blob_{image_uuid}.json", ts

    def to_model_frame(self, points: np.ndarray, blob_key: str, timestamp: int) -> np.ndarray | None:
        """(N, 2) post-scan pixels -> pixels of the model's 256 prescan frame; None if a point doesn't convert.

        Clipped to the frame, as training clips its boxes (isonoai's mask_box: [0, 255]): points past the scan
        area convert to points outside the prescan image. Linear probes only.
        """
        key, images_json, (pre_w, pre_h) = self.load_metadata(blob_key, timestamp)
        with self.converter_lock:
            if self.converter_key != key:
                self.convert_utils.loadImageMetadata(images_json)
                self.converter_key = key
            converted = [
                convert_point_to_prescan(point, timestamp, LS_WIDTH, LS_HEIGHT, self.convert_utils) for point in points
            ]
        if any(p is None or not np.all(np.isfinite(p)) for p in converted):
            return None
        return np.clip(np.array(converted, dtype=np.float32) / [pre_w, pre_h] * INPUT_SIZE, 0, INPUT_SIZE - 1)

    def convert_box_to_prescan(self, corners: np.ndarray, blob_key: str, timestamp: int) -> np.ndarray | None:
        """A box's post-scan corners (box_corners) -> the box in the model's 256 frame, as (x0, y0, x1, y1);
        None if it doesn't convert. min/max, since the conversion may flip or swap the axes."""
        points = self.to_model_frame(corners, blob_key, timestamp)
        if points is None:
            return None
        (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
        if x1 <= x0 or y1 <= y0:
            return None
        return np.array([x0, y0, x1, y1], dtype=np.float32)

    def annotation_pairs(self, task_data: dict, image_url: str, results: list[dict], trainer: OnlineTrainer) -> list:
        """A submitted annotation's training pairs: one per mapped label, all its polygons on the image combined
        (data.py). Polygons that don't convert to the prescan frame are left out."""
        prescan_key, blob_key, ts = self.locate(task_data, image_url)
        polygons = defaultdict(list)
        for result in results:
            if result["type"] != "polygonlabels":
                continue
            name = label_name(result["value"]["polygonlabels"][0])
            if name not in trainer.label_structure:
                continue
            postscan = np.array(result["value"]["points"]) / 100 * [LS_WIDTH, LS_HEIGHT]
            points = self.to_model_frame(postscan, blob_key, ts)
            if points is not None:
                polygons[name].append(points.tolist())
        return [
            {"image": prescan_key, "structure": trainer.label_structure[name], "polygons": p}
            for name, p in polygons.items()
        ]

    def convert_mask_to_postscan(self, logits: np.ndarray, blob_key: str, timestamp: int) -> np.ndarray:
        """(256, 256) logits from the model's frame -> binary mask in the post-scan frame."""
        key, images_json, (pre_w, pre_h) = self.load_metadata(blob_key, timestamp)
        with self.converter_lock:
            if self.converter_key != key:
                self.convert_utils.loadImageMetadata(images_json)
                self.converter_key = key
            postscan_mask = postprocess_mask(logits, timestamp, pre_w, pre_h, self.convert_utils)
        return postscan_mask


def ls_get(path: str, **params) -> dict:
    """GET from the Label Studio API at $LABEL_STUDIO_URL, with $LABEL_STUDIO_API_KEY."""
    url = os.environ["LABEL_STUDIO_URL"].rstrip("/") + path
    response = requests.get(url, params, headers={"Authorization": f"Token {os.environ['LABEL_STUDIO_API_KEY']}"})
    response.raise_for_status()
    return response.json()


def annotated_tasks(project_id: int) -> list[dict]:
    """Every task of the project with its annotations, a page at a time (app.heartex.com has no export endpoint)."""
    tasks, page = [], 1
    while True:
        try:
            batch = ls_get("/api/tasks", project=project_id, fields="all", page_size=100, page=page)["tasks"]
        except requests.HTTPError as e:  # past the last page
            if e.response.status_code == 404:
                return tasks
            raise
        tasks += [task for task in batch if task["annotations"]]
        page += 1


class NewModel(LabelStudioMLBase):
    """TSAM: isonoai's box-only SAM, prompted with a box.

    label-studio-ml creates a new instance per request, so the handler that should
    outlive a request is a class attribute shared by all instances.
    """

    # Set once at startup by main.py (`NewModel.configure`), before the server takes requests.
    handler: TSAMHandler | None = None
    trainer: OnlineTrainer | None = None

    @classmethod
    def configure(cls, config: Config) -> None:
        """Load the model (the run's own, when continuing one) and start online training; every instance shares them.
        A new run first backfills."""
        cls.handler = TSAMHandler(config.continue_run_id or config.sam.base_run_id, EMBEDDING_CACHE_SIZE)
        cls.trainer = OnlineTrainer(config, cls.handler.sam, lambda key: cls.handler.preprocess_img(key)[0])
        if not config.continue_run_id:
            cls.backfill(config.trainer.backfill_limit)

    @classmethod
    def backfill(cls, limit: int | None) -> None:
        """Feed the project's latest `limit` annotations (None: all) through fit(), oldest first, as if each had just
        been submitted. This queues them; the updates train on them in the background."""
        project_id = cls.trainer.config.project_id
        project = ls_get(f"/api/projects/{project_id}")
        model = cls(project_id=str(project_id), label_config=project["label_config"])
        submitted = sorted(
            ((annotation, task) for task in annotated_tasks(project_id) for annotation in task["annotations"]),
            key=lambda pair: pair[0]["created_at"],
        )
        backfill = submitted if limit is None else submitted[max(len(submitted) - limit, 0) :]
        logger.info(f"Backfilling {len(backfill)} of {len(submitted)} annotations of project {project['title']}")
        for annotation, task in backfill:
            model.fit("ANNOTATION_CREATED", {"annotation": annotation, "task": task})

    def setup(self):
        """Report the online-training run as the model version, so predictions say which model made them."""
        self.set("model_version", f"tsam-{self.trainer.run_id}")

    def fit(self, event: str, data: dict, **kwargs) -> None:
        """Queue a submitted annotation's labels for online training (Label Studio's webhook)."""
        if event not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
            return
        if int(self.project_id) != self.trainer.config.project_id:
            return
        _, _, value = self.get_first_tag_occurence("PolygonLabels", "Image")
        task_data = data["task"]["data"]
        pairs = self.handler.annotation_pairs(task_data, task_data[value], data["annotation"]["result"], self.trainer)
        self.trainer.add(pairs)

    def predict(self, tasks: list[dict], context: dict | None = None, **kwargs) -> ModelResponse:
        """Returns the mask for the latest box drawn on this image."""
        from_name, to_name, value = self.get_first_tag_occurence("BrushLabels", "Image")
        if not context or not context.get("result"):
            logger.warning("No context! -- cannot make a prediction.")
            return ModelResponse(predictions=[])  # no interaction yet

        # Label Studio sends every smart box with the drawn one's label; the latest is the one just drawn.
        boxes = [ctx for ctx in context["result"] if ctx["type"] == "rectanglelabels"]
        if not boxes:
            logger.info("TSAM's box-only model needs a box prompt -- clicks aren't supported.")
            return ModelResponse(predictions=[])
        box = boxes[-1]

        task = tasks[0]
        prescan_key, blob_key, ts = self.handler.locate(task["data"], task["data"][value])

        # Compute embeddings for the image using the handler.
        embeddings = self.handler.embed(prescan_key)

        # The box in Label Studio's percent coordinates -> (x0, y0, x1, y1) pixels of the model's 256 frame.
        model_box = self.handler.convert_box_to_prescan(box_corners(box["value"]), blob_key, ts)
        if model_box is None:
            logger.warning(f"The box doesn't map onto the scan area of {prescan_key}; no prediction.")
            return ModelResponse(predictions=[])
        label = box["value"]["rectanglelabels"][0]

        # The trainer's live decoder, with the box's structure (None if its label isn't mapped: no offset)
        structure = self.trainer.label_structure.get(label_name(label))
        logits, iou = self.trainer.predict(embeddings, tf.constant(model_box[None]), structure)
        # Smooth and keep the largest component.
        logits = refine_logits(logits[0].astype(np.float32))

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
