"""online.py: the losses, the decoder, and the trainer end to end against a local MLflow: baseline, updates, replay,
acceptance, rollback, queueing, checkpoint and resume."""

import json
from pathlib import Path

import cv2
import mlflow
import numpy as np
import pytest
import tensorflow as tf
from conftest import FIBROUS, SKIN, annotation, load_image, tiny_sam

import tsam.online
from tsam.online import OnlineTrainer, dice_loss, mask_iou
from tsam.utils.sam import EXCLUDED_LOGIT


def weights(variables) -> list[np.ndarray]:
    return [v.numpy().copy() for v in variables]


def same(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def artifacts(run_id: str, path: str) -> Path:
    return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path))


@pytest.fixture
def trainer(config, local_mlflow) -> OnlineTrainer:
    return OnlineTrainer(config, tiny_sam(), load_image)


def test_dice_counts_each_instance_equally():
    masks = np.zeros((2, 256, 256), np.float32)
    masks[0, :100, :100] = 1  # large, predicted exactly
    masks[1, 200:205, 200:205] = 1  # small, missed entirely
    logits = np.where(masks[:1] > 0, 20.0, -20.0)
    logits = np.concatenate([logits, np.full((1, 256, 256), -20.0)])
    assert dice_loss(tf.constant(logits, tf.float32), tf.constant(masks)).numpy() == pytest.approx(0.5, abs=0.05)


def test_mask_iou_thresholds_logits_at_zero():
    masks = np.zeros((1, 256, 256), np.float32)
    masks[0, :10, :10] = 1
    logits = np.full((1, 256, 256), -1.0, np.float32)
    logits[0, :10, :5] = 1  # half the mask
    assert mask_iou(tf.constant(logits), tf.constant(masks)).numpy() == pytest.approx([0.5])


def test_decoder_is_the_sam_pass_after_the_image_encoder(trainer):
    rng = np.random.default_rng(0)
    images = tf.constant(rng.random((2, 256, 256, 1)), tf.float32)
    boxes = tf.constant([[10, 10, 80, 90], [100, 120, 200, 140]], tf.float32)
    structures = tf.constant([SKIN, FIBROUS])
    accepted = np.zeros((2, 256, 256), np.int32)
    accepted[0, 50:70, 50:70] = FIBROUS
    accepted = tf.constant(accepted)
    expected = trainer.sam({"images": images, "boxes": boxes, "structures": structures, "accepted": accepted})
    logits, iou = trainer.live(trainer.encoder(images), boxes, structures, accepted)
    np.testing.assert_allclose(logits, expected["mask"], atol=1e-4)
    np.testing.assert_allclose(iou, expected["iou"], atol=1e-4)
    assert np.all(logits.numpy()[accepted.numpy() > 0] == EXCLUDED_LOGIT)


def test_a_model_with_other_structures_is_refused(config, local_mlflow):
    sam = tiny_sam()
    sam.prompt_encoder.structures = ["Anything", "Skin"]
    with pytest.raises(ValueError, match="structures"):
        OnlineTrainer(config, sam, load_image)


def test_new_run_logs_baseline_config_and_base_examples(trainer, local_mlflow):
    assert trainer.step == 0 and len(trainer.test_set["mask"]) == 9
    run = local_mlflow.get_run(trainer.run_id)
    assert run.data.metrics["test_iou"] == pytest.approx(trainer.best)
    assert {"test_iou/Skin", "test_iou/Fibrous"} <= set(run.data.metrics)
    assert artifacts(trainer.run_id, "config.yaml").exists()
    examples = sorted(artifacts(trainer.run_id, "examples/base").glob("*.png"))
    assert len(examples) == 3  # every test image, there being fewer than 20
    assert cv2.imread(str(examples[0])).shape == (512, 1024, 3)  # labels beside predictions


def test_update_takes_fixed_steps_and_mixes_in_replay(trainer, monkeypatch):
    batches = []
    train_step = trainer._train_step
    monkeypatch.setattr(trainer, "_train_step", lambda batch: batches.append(batch) or train_step(batch))

    first = [annotation("a", SKIN, FIBROUS, FIBROUS), annotation("b", SKIN)]
    trainer._update(first)
    assert len(batches) == 3 and all(len(b) == 4 for b in batches)  # steps_per_update, minibatch_size
    assert all(any(a is f for f in first) for b in batches for a in b)  # nothing to replay yet

    batches.clear()
    second = [annotation("c", FIBROUS)]
    trainer._update(second)
    assert len(batches) == 3
    assert all(sum(any(a is n for n in second) for a in b) == 2 for b in batches)  # replay ratio 1: half replayed
    assert trainer.step == 2 and trainer.buffer == first + second


def test_accepted_update_goes_live_and_is_checkpointed(trainer, local_mlflow):
    before = weights(trainer.live.variables)
    trainer._update([annotation("a", SKIN, FIBROUS), annotation("b", FIBROUS)])
    assert not same(before, weights(trainer.live.variables))
    assert same(weights(trainer.live.variables), weights(trainer.train.variables))

    assert local_mlflow.get_run(trainer.run_id).data.metrics["accepted"] == 1
    assert (artifacts(trainer.run_id, "model") / "MLmodel").exists()
    checkpoint = artifacts(trainer.run_id, "checkpoint")
    assert json.loads((checkpoint / "progress.json").read_text())["step"] == 1
    assert len(json.loads((checkpoint / "buffer.json").read_text())) == 2
    assert len(list(artifacts(trainer.run_id, "examples/deployed").glob("*.png"))) == 3


def test_rejected_update_rolls_back(config, local_mlflow):
    config.test.acceptance_threshold = 1.0  # no update can gain a whole point of IoU
    trainer = OnlineTrainer(config, tiny_sam(), load_image)
    live, train = weights(trainer.live.variables), weights(trainer.train.variables)
    trainer._update([annotation("a", SKIN)])
    assert same(live, weights(trainer.live.variables))
    assert same(train, weights(trainer.train.variables))
    assert local_mlflow.get_run(trainer.run_id).data.metrics["accepted"] == 0
    assert len(trainer.buffer) == 1  # it is still replayed later


def test_each_update_takes_samples_per_update_annotations(trainer, monkeypatch):
    class Immediately:  # runs the update in place of starting its thread
        def __init__(self, target, args, daemon):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    updates = []
    monkeypatch.setattr(tsam.online.threading, "Thread", Immediately)
    monkeypatch.setattr(trainer, "_update", updates.append)  # leaves `updating` set, as a running update does
    a = [annotation(str(i), SKIN) for i in range(5)]

    trainer.add(a[0])
    assert updates == []  # one annotation of the two an update needs
    trainer.add(a[1])
    assert updates == [[a[0], a[1]]]
    trainer.add(a[2])
    trainer.add(a[3])
    trainer.add(a[4])
    trainer.add(annotation("none"))  # an annotation with no labels in a structure isn't queued
    assert len(updates) == 1  # an update is running
    trainer.updating = False
    trainer._start_update()
    assert updates[1] == [a[2], a[3]] and trainer.queue == [a[4]]


def test_resume_restores_the_run(trainer, config):
    from tsam.model import load_sam

    trainer._update([annotation("a", SKIN, FIBROUS), annotation("b", FIBROUS)])
    config.continue_run_id = trainer.run_id
    resumed = OnlineTrainer(config, load_sam(trainer.run_id), load_image)
    assert (resumed.run_id, resumed.step, resumed.best) == (trainer.run_id, 1, trainer.best)
    assert resumed.buffer == json.loads(json.dumps(trainer.buffer))
    assert same(weights(resumed.live.variables), weights(trainer.live.variables))
    assert same(weights(resumed.train.variables), weights(trainer.live.variables))
