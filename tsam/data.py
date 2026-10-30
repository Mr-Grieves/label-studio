"""Training and test data for online training, all in the model's 256 x 256 prescan frame.

A pair is one image and one label's mask (all its polygons on that image combined), as isonoai's SAMDataset trains
T-SAM on. Training pairs are records {"image": prescan S3 key, "structure": index, "polygons": [[[x, y], ...], ...]}
that are rasterized when used; test pairs come from a tfrecord of `pre_scan_image` and `{label}/pre_scan_mask`s.
"""

import os
import re
from pathlib import Path
from typing import Any

import cv2
import keras
import numpy as np
import tensorflow as tf

INPUT_SIZE = 256  # the model's input frame (as trained); boxes are in its pixels
MIN_PIXELS = 10  # smaller masks are left out, as isonoai's SAMDataset does


def label_name(label: str) -> str:
    """A Label Studio label without its tool suffix: "ATFL [SAM]" -> "ATFL"."""
    return re.sub(r"\s*\[[^\]]*\]$", "", label)


def structure_ids(mapping: dict[str, list[str]]) -> dict[str, int]:
    """Label name -> the index of its structure, in the config's order."""
    return {name: i for i, names in enumerate(mapping.values()) for name in names}


def rasterize(polygons: list) -> np.ndarray:
    """Polygons in the 256 frame -> their combined (256, 256) uint8 mask."""
    mask = np.zeros((INPUT_SIZE, INPUT_SIZE), np.uint8)
    return cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in polygons], 1)


def mask_box(mask: tf.Tensor, jitter: float = 0.0) -> tf.Tensor:
    """(B, 256, 256) masks, none empty -> (B, 4) boxes (x0, y0, x1, y1): the pixels at each mask's extremes,
    each coordinate shifted by up to `jitter` pixels, then clipped to the frame. Ported from isonoai's sam_utils."""
    present = mask > 0.5

    def extent(axis_present: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        index = tf.range(tf.shape(axis_present)[1], dtype=tf.float32)
        lo = tf.reduce_min(tf.where(axis_present, index, float(INPUT_SIZE)), axis=1)
        hi = tf.reduce_max(tf.where(axis_present, index, -1.0), axis=1)
        return lo, hi

    (y0, y1), (x0, x1) = extent(tf.reduce_any(present, axis=2)), extent(tf.reduce_any(present, axis=1))
    box = tf.stack([x0, y0, x1, y1], axis=-1)
    box = tf.clip_by_value(box + tf.random.uniform(tf.shape(box), -jitter, jitter), 0.0, INPUT_SIZE - 1.0)
    # Jitter can cross a thin box's corners over: keep (x0, y0) the top-left
    return tf.concat([tf.minimum(box[:, :2], box[:, 2:]), tf.maximum(box[:, :2], box[:, 2:])], axis=-1)


def build_augmentation(specs: list[dict[str, dict[str, Any]]]) -> list[keras.layers.Layer]:
    """The config's augmentation, e.g. [{"RandomFlip": {"mode": "vertical"}}], as keras.layers layers."""
    return [getattr(keras.layers, name)(**kwargs) for spec in specs for name, kwargs in spec.items()]


def augment(layers: list[keras.layers.Layer], images: tf.Tensor, masks: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Augment (B, 256, 256, 1) images and their masks together, each pair drawing its own transform."""
    data = {"images": images, "segmentation_masks": masks}
    for layer in layers:
        data = layer(data, training=True)
    # Geometric layers interpolate the mask
    return data["images"], tf.cast(data["segmentation_masks"] > 0.5, tf.float32)


def tf_decode_image(blob: tf.Tensor, channels: int) -> tf.Tensor:
    """A PNG- or JPEG-encoded image -> uint8 (H, W, channels), decoded as isonoai decodes training images."""
    return tf.cond(
        tf.equal(tf.strings.substr(blob, 0, 4), b"\x89PNG"),
        lambda: tf.io.decode_png(blob, channels=channels),
        lambda: tf.io.decode_jpeg(blob, channels=channels, dct_method="INTEGER_ACCURATE"),
    )


def to_model_frame(image: tf.Tensor) -> tf.Tensor:
    """A uint8 (H, W, 1) prescan image -> (256, 256, 1) float32 in [0, 1], as T-SAM was trained."""
    return tf.image.resize(tf.cast(image, tf.float32) / 255.0, (INPUT_SIZE, INPUT_SIZE), method="bilinear")


def read_test_set(paths: list[Path], label_structure: dict[str, int]) -> dict[str, np.ndarray]:
    """The test pairs of every mapped label in the tfrecords at `paths` (relative ones under $DATA_DIR).

    Returns:
        `images` (N, 256, 256, 1), and per pair: `image` (its index in `images`), `mask` (256, 256) uint8, and
        `structure`.
    """
    images, pairs = [], []
    for path in paths:
        for raw in tf.data.TFRecordDataset(str(Path(os.getenv("DATA_DIR", "/data")) / path)):
            features = tf.train.Example.FromString(raw.numpy()).features.feature
            for key, feature in features.items():
                label = key.removesuffix("/pre_scan_mask")
                if label == key or label not in label_structure or not feature.bytes_list.value:
                    continue
                mask = tf.io.decode_png(feature.bytes_list.value[0], channels=1)
                mask = tf.image.resize(mask, (INPUT_SIZE, INPUT_SIZE), method="nearest")[..., 0].numpy() > 0
                if mask.sum() >= MIN_PIXELS:
                    pairs.append((len(images), mask.astype(np.uint8), label_structure[label]))
            if pairs and pairs[-1][0] == len(images):  # this record has a pair
                images.append(to_model_frame(tf_decode_image(features["pre_scan_image"].bytes_list.value[0], 1)))
    image, mask, structure = zip(*pairs)
    return {
        "images": np.stack(images),
        "image": np.array(image),
        "mask": np.stack(mask),
        "structure": np.array(structure),
    }
