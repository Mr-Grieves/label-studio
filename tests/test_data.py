"""data.py: labels, instances, boxes, augmentation, drawing and the test set."""

import numpy as np
import tensorflow as tf
from conftest import blob, square

from tsam.data import (
    augment,
    build_augmentation,
    components,
    draw,
    instances,
    label_name,
    mask_box,
    rasterize,
    read_test_set,
    structure_ids,
)


def test_label_name_strips_tool_suffix():
    assert label_name("ATFL [SAM]") == "ATFL"
    assert label_name("Skin") == "Skin"


def test_structure_ids_map_every_label_name():
    assert structure_ids({"Skin": ["Skin"], "Tendon": ["Tendon", "FDT"]}) == {"Skin": 0, "Tendon": 1, "FDT": 1}


def test_components_split_a_mask_and_drop_specks():
    mask = blob(blob(blob(np.zeros((256, 256), np.uint8), 10, 10, 10), 100, 100, 10), 200, 200, 2)
    parts = components(mask)
    assert [int(p.sum()) for p in parts] == [100, 100]
    assert all(p.dtype == np.uint8 for p in parts)


def test_instances_group_overlapping_polygons_and_drop_specks():
    overlapping = [square(10, 10, 20), square(20, 20, 20)]
    separate, speck = square(100, 100, 20), square(200, 200, 1)
    groups = instances([*overlapping, separate, speck])
    assert sorted(map(len, groups)) == [1, 2]
    assert all(len(components(rasterize(group))) == 1 for group in groups)


def test_mask_box_is_the_mask_extremes():
    mask = np.zeros((1, 256, 256), np.float32)
    mask[0, 10:20, 30:50] = 1
    assert mask_box(tf.constant(mask)).numpy().tolist() == [[30, 10, 49, 19]]


def test_jittered_box_stays_ordered_and_in_frame():
    mask = np.zeros((64, 256, 256), np.float32)
    mask[:, 0:3, 250:252] = 1  # thin, at the frame's edge
    box = mask_box(tf.constant(mask), jitter=10).numpy()
    assert np.all(box[:, :2] <= box[:, 2:]) and box.min() >= 0 and box.max() <= 255


def test_augmentation_keeps_images_and_masks_aligned():
    masks = np.zeros((8, 256, 256, 1), np.float32)
    masks[:, 60:140, 80:200] = 1
    layers = build_augmentation(
        [
            {"RandomFlip": {"mode": "vertical"}},
            {"RandomTranslation": {"height_factor": 0.1, "width_factor": 0.1, "fill_mode": "constant"}},
            {"RandomZoom": {"height_factor": 0.1, "fill_mode": "constant"}},
        ]
    )
    images, augmented = augment(layers, tf.constant(masks), tf.constant(masks))  # the image is its own mask
    image = images.numpy() > 0.5
    mask = augmented.numpy() > 0.5
    assert (image & mask).sum() / (image | mask).sum() > 0.95


def test_draw_rotates_clockwise():
    image = np.full((256, 256, 1), 0.5, np.float32)
    mask = np.zeros((1, 256, 256), np.uint8)
    mask[0, 100:120, 0:20] = 1  # rows 100-119 at the left edge -> the top, columns 136-155, once rotated
    box = mask_box(tf.constant(mask, tf.float32)).numpy()
    canvas = draw(image, mask, box, np.array([0]), ["Skin", "Tendon"])
    colored = lambda pixel: len(set(pixel.tolist())) > 1  # the image is gray; masks and boxes are colored
    assert canvas.shape == (512, 512, 3)
    assert colored(canvas[20, 290])  # inside the rotated mask, at twice the scale
    assert not colored(canvas[220, 20])  # where the mask was before rotating


def test_test_set_has_one_pair_per_component(test_tfrecord):
    t = read_test_set([test_tfrecord], {"Skin": 0, "Tendon": 1})
    assert t["images"].shape == (3, 256, 256, 1)
    assert np.bincount(t["structure"]).tolist() == [3, 6]  # 1 Skin and 2 Tendons per image; speck and Unmapped out
    assert all(len(components(mask)) == 1 for mask in t["mask"])
    assert t["image"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
