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
import logging
import os
import threading
from collections import OrderedDict
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
import tensorflow as tf
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
INPUT_SIZE = 256  # the model's input frame (as trained); clicks are in its pixels
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", 16))
LOGITS_CACHE_SIZE = int(os.getenv("LOGITS_CACHE_SIZE", 256))

ENCODER = tf.saved_model.load(os.path.join(MODEL_DIR, "encoder"))
DECODER = tf.saved_model.load(os.path.join(MODEL_DIR, "decoder"))


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


EMBEDDINGS = LRUCache(EMBEDDING_CACHE_SIZE)   # (task id, image url) -> (embeddings, H, W)
LOGITS = LRUCache(LOGITS_CACHE_SIZE)          # (task id, image url, clicks) -> that prediction's logits


def preprocess(path: str):
    """Image file -> (1, 256, 256, 1) float32 in [0, 1], plus its original (H, W)."""
    image = Image.open(path).convert("L")
    width, height = image.size
    pixels = np.asarray(image, dtype=np.float32)[None, ..., None] / 255.0
    return tf.image.resize(pixels, (INPUT_SIZE, INPUT_SIZE), method="bilinear"), height, width


class NewModel(LabelStudioMLBase):
    """effsam: isonoai's SAM, prompted with clicks."""

    def _embed(self, image_url: str, task_id) -> tuple:
        key = (task_id, image_url)
        cached = EMBEDDINGS.get(key)
        if cached is None:
            images, height, width = preprocess(self.get_local_path(image_url, task_id=task_id))
            cached = (ENCODER.serve([images]), height, width)
            EMBEDDINGS.put(key, cached)
        return cached

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Returns the mask for the clicks made so far on this image."""
        from_name, to_name, value = self.get_first_tag_occurence("BrushLabels", "Image")
        if not context or not context.get("result"):
            return ModelResponse(predictions=[])  # no interaction yet

        # Clicks in Label Studio's percent coordinates -> pixels of the model's 256 frame.
        clicks, label = [], None
        for ctx in context["result"]:
            if ctx["type"] != "keypointlabels":
                continue
            x = ctx["value"]["x"] / 100 * INPUT_SIZE
            y = ctx["value"]["y"] / 100 * INPUT_SIZE
            clicks.append((x, y, 1.0 if ctx.get("is_positive", True) else 0.0))
            label = ctx["value"]["keypointlabels"][0]
        if not clicks:
            logger.info("effsam only uses click prompts -- add KeyPointLabels clicks instead of a box.")
            return ModelResponse(predictions=[])

        task = tasks[0]
        image_url = task["data"][value]
        embeddings, height, width = self._embed(image_url, task.get("id"))

        # Mask loopback: reuse the previous prediction if these clicks extend its clicks by one.
        clicks = tuple(clicks)
        previous = LOGITS.get((task.get("id"), image_url, clicks[:-1])) if len(clicks) > 1 else None
        out = DECODER.serve({
            "embeddings": embeddings,
            "points": tf.constant([[c[:2] for c in clicks]], dtype=tf.float32),   # (1, N, 2)
            "labels": tf.constant([[c[2] for c in clicks]], dtype=tf.float32),    # (1, N)
            "pred_mask": previous if previous is not None else tf.zeros((1, 4 * embeddings.shape[1], 4 * embeddings.shape[2])),
            "has_mask": tf.constant([0.0 if previous is None else 1.0]),
        })
        logits = out["logits"]
        LOGITS.put((task.get("id"), image_url, clicks), logits)

        # Low-res logits -> mask at the image's original size.
        full = tf.image.resize(logits[..., None], (height, width), method="bilinear")[0, ..., 0]
        mask = (full.numpy() > 0).astype(np.uint8)
        score = float(np.clip(out["iou"].numpy()[0], 0.0, 1.0))  # the IoU head's estimate of this mask's quality

        return ModelResponse(predictions=[{
            "result": [{
                "id": str(uuid4())[:4],
                "from_name": from_name,
                "to_name": to_name,
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {"format": "rle", "rle": brush.mask2rle(mask * 255), "brushlabels": [label]},
                "score": score,
                "type": "brushlabels",
                "readonly": False,
            }],
            "model_version": self.get("model_version"),
            "score": score,
        }])
