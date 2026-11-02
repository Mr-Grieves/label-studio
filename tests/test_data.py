"""data.py: instances, accepted masks, boxes, augmentation, drawing and the test set."""

import numpy as np
import tensorflow as tf
from conftest import FIBROUS, SKIN, blob, square

from tsam.data import (
    augment,
    build_augmentation,
    components,
    draw,
    draw_accepted,
    instances,
    mask_box,
    rasterize,
    read_test_set,
)


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


def test_accepted_masks_are_a_random_share_of_the_others():
    around = blob(np.zeros((256, 256), np.uint8), 0, 0, 100)  # the drawn instance lies inside it
    drawn = blob(np.zeros((256, 256), np.uint8), 20, 20, 20)
    apart = blob(np.zeros((256, 256), np.uint8), 150, 150, 20)
    masks, structures = np.stack([around, drawn, apart]) // 255, np.array([SKIN, FIBROUS, FIBROUS])
    rng = np.random.default_rng(0)
    draws = [draw_accepted(masks, structures, 1, rng) for _ in range(200)]
    assert all(not d[masks[0] > 0].any() for d in draws)  # never the one around it, nor itself
    taken = [d[masks[2] > 0] for d in draws]
    assert all(np.all(t == 0) or np.all(t == FIBROUS) for t in taken)  # its structure, where it is
    assert 50 < sum(t.any() for t in taken) < 150  # accepted about half the time: P(u < p), u and p uniform


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
    t = read_test_set([test_tfrecord])
    assert t["images"].shape == (3, 256, 256, 1)
    # 1 Skin and 2 Tendons per image; the speck and the label in no structure are left out
    assert sorted(t["structure"].tolist()) == [SKIN] * 3 + [FIBROUS] * 6
    assert all(len(components(mask)) == 1 for mask in t["mask"])
    assert t["image"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_test_set_accepted_masks_are_the_frames_others_and_fixed(test_tfrecord):
    t = read_test_set([test_tfrecord])
    assert t["accepted"].shape == t["mask"].shape
    assert not np.any((t["accepted"] > 0) & (t["mask"] > 0))  # the instances don't overlap: never its own pixels
    assert set(np.unique(t["accepted"])) <= {0, SKIN, FIBROUS} and t["accepted"].any()
    assert np.array_equal(read_test_set([test_tfrecord])["accepted"], t["accepted"])
