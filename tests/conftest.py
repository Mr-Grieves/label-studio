"""Shared fixtures: a tiny SAM built from the real classes, a synthetic test tfrecord, a config, and a local MLflow.
Everything runs offline: no S3, Label Studio or MLflow server."""

import os
import tempfile
import zlib

# label_studio_ml creates its cache.db at import, in $MODEL_DIR: keep it out of the repo
os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp())

import cv2
import keras
import mlflow
import numpy as np
import pytest
import tensorflow as tf

from tsam.config import Config
from tsam.utils.sam import SAM, MaskDecoder, PromptEncoder

DIM = 32  # small, for speed; the real T-SAM's is 256


def tiny_sam() -> SAM:
    """A SAM with the real prompt encoder and mask decoder, and a one-layer image encoder: 256 frame, 64 grid."""
    encoder = keras.Sequential([keras.Input((256, 256, 1)), keras.layers.Conv2D(DIM, 4, strides=4)])
    return SAM(encoder, PromptEncoder(dim=DIM), MaskDecoder(dim=DIM, heads=2))


def blob(mask: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    mask[y : y + size, x : x + size] = 255
    return mask


def png(array: np.ndarray) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[cv2.imencode(".png", array)[1].tobytes()]))


@pytest.fixture
def test_tfrecord(tmp_path):
    """3 images, each with one Skin instance, two separate Tendon instances (and a speck under MIN_PIXELS), and an
    unmapped label: 9 test pairs."""
    path = tmp_path / "test.tfrecord"
    rng = np.random.default_rng(0)
    with tf.io.TFRecordWriter(str(path)) as writer:
        for _ in range(3):
            image = rng.integers(0, 255, (256, 256), np.uint8)
            skin = blob(np.zeros((256, 256), np.uint8), 0, 0, 40)
            tendon = blob(blob(blob(np.zeros((256, 256), np.uint8), 100, 100, 30), 180, 60, 20), 250, 250, 2)
            other = blob(np.zeros((256, 256), np.uint8), 50, 200, 30)
            features = {
                "pre_scan_image": png(image),
                "Skin/pre_scan_mask": png(skin),
                "Tendon/pre_scan_mask": png(tendon),
                "Unmapped/pre_scan_mask": png(other),
            }
            writer.write(tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString())
    return path


@pytest.fixture
def config(test_tfrecord) -> Config:
    return Config.model_validate(
        {
            "project_id": 1,
            "description": "test",
            "experiment_name": "test",
            "run_name": "test",
            "version": "0",
            "structures": {"conditioning": True, "mapping": {"Skin": ["Skin"], "Tendon": ["Tendon", "FDT"]}},
            "model_config": {"base_run_id": "base"},
            "optimizer_config": {"name": "adamw", "learning_rate": 1e-4},
            "trainer_config": {
                "samples_per_update": 2,
                "backfill_limit": None,
                "minibatch_size": 4,
                "steps_per_update": 3,
                "box_jitter": 2,
                "replay": {"ratio": 1.0},
                "augmentation": [
                    {"RandomFlip": {"mode": "vertical"}},
                    {"RandomTranslation": {"height_factor": 0.05, "width_factor": 0.05}},
                ],
                "loss_function": [{"key": "mask", "name": "dice"}, {"key": "iou", "name": "mse", "weight": 0.1}],
            },
            "test_config": {"dataset_paths": [str(test_tfrecord)], "batch_size": 4, "acceptance_threshold": -1.0},
        }
    )


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """A throwaway MLflow in tmp_path; its artifacts land in the working directory, so that's tmp_path too."""
    monkeypatch.chdir(tmp_path)
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    yield mlflow.MlflowClient()
    mlflow.set_tracking_uri(previous)


def load_image(key: str) -> tf.Tensor:
    """A fixed random prescan image per key, as the trainer's load_image gives: (256, 256, 1) float32 in [0, 1]."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return tf.constant(rng.random((256, 256, 1)), tf.float32)


def square(x: float, y: float, size: float) -> list:
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]


def pair(image: str, structure: int, x: float = 60, y: float = 60, size: float = 40) -> dict:
    return {"image": image, "structure": structure, "polygons": [square(x, y, size)]}
