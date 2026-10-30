"""Training and test data for online training, all in the model's 256 x 256 prescan frame.

A pair is one image and one instance of a structure on it: a connected component of its label's mask, as a user boxes
it in Label Studio. Training pairs are records {"image": prescan S3 key, "structure": index, "polygons":
[[[x, y], ...], ...]}, the polygons forming that component, rasterized when used; test pairs are the components of the
`{label}/pre_scan_mask`s in a tfrecord of them and `pre_scan_image`.
"""

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import keras
import numpy as np
import tensorflow as tf

INPUT_SIZE = 256  # the model's input frame (as trained); boxes are in its pixels
MIN_PIXELS = 10  # smaller components are left out, as isonoai's SAMDataset leaves out masks


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


def components(mask: np.ndarray) -> list[np.ndarray]:
    """A (256, 256) mask's 8-connected components of at least MIN_PIXELS, each as a uint8 mask."""
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return [c for c in ((labels == i).astype(np.uint8) for i in range(1, n)) if c.sum() >= MIN_PIXELS]


def instances(polygons: list) -> list[list]:
    """One label's polygons on an image, grouped by the component of their combined mask that each lies in: a group
    per instance of the structure. Components under MIN_PIXELS are left out."""
    _, labels = cv2.connectedComponents(rasterize(polygons), connectivity=8)
    groups = defaultdict(list)
    for polygon in polygons:
        pixels = labels[rasterize([polygon]) > 0]
        if pixels.size:
            groups[np.bincount(pixels).argmax()].append(polygon)
    return [group for group in groups.values() if rasterize(group).sum() >= MIN_PIXELS]


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


def draw(image: np.ndarray, masks: np.ndarray, boxes: np.ndarray, structures: np.ndarray, names: list) -> np.ndarray:
    """A (256, 256, 1) prescan image in [0, 1], twice as large, with (N, 256, 256) masks filled and outlined, and their
    (N, 4) boxes and structure names, each structure in its own color -> (512, 512, 3) uint8 BGR. Rotated 90 degrees
    clockwise, to show the prescan frame upright."""
    size = 2 * INPUT_SIZE
    image, masks = np.ascontiguousarray(np.rot90(image, -1)), np.ascontiguousarray(np.rot90(masks, -1, axes=(1, 2)))
    x0, y0, x1, y1 = np.moveaxis(boxes, -1, 0)
    boxes = np.stack([INPUT_SIZE - 1 - y1, x0, INPUT_SIZE - 1 - y0, x1], axis=-1)
    canvas = cv2.cvtColor(cv2.resize(np.uint8(image[..., 0] * 255), (size, size)), cv2.COLOR_GRAY2BGR)
    palette = cv2.applyColorMap(np.uint8((np.arange(len(names)) + 0.5) * 255 / len(names))[:, None], cv2.COLORMAP_TURBO)
    masks = [cv2.resize(np.uint8(m), (size, size), interpolation=cv2.INTER_NEAREST) for m in masks]
    fill = canvas.copy()
    for mask, s in zip(masks, structures):
        fill[mask > 0] = palette[s, 0]
    canvas = cv2.addWeighted(canvas, 0.6, fill, 0.4, 0)
    for mask, box, s in zip(masks, np.round(2 * boxes).astype(int), structures):
        color = palette[s, 0].tolist()
        cv2.drawContours(canvas, cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1, color, 1)
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), color, 1)
        corner = (box[0], max(box[1] - 3, 10))
        cv2.putText(canvas, names[s], corner, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas


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
    """The test pairs of every mapped label in the tfrecords at `paths` (relative ones under $DATA_DIR): one per
    component of its mask.

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
                mask = tf.image.resize(mask, (INPUT_SIZE, INPUT_SIZE), method="nearest")[..., 0].numpy()
                pairs += [(len(images), c, label_structure[label]) for c in components(mask)]
            if pairs and pairs[-1][0] == len(images):  # this record has a pair
                images.append(to_model_frame(tf_decode_image(features["pre_scan_image"].bytes_list.value[0], 1)))
    image, mask, structure = zip(*pairs)
    return {
        "images": np.stack(images),
        "image": np.array(image),
        "mask": np.stack(mask),
        "structure": np.array(structure),
    }
