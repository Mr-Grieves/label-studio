"""Online training of T-SAM on one Label Studio project, as its labels are submitted (see online-training.md).

The image encoder stays frozen, shared with serving. Training updates a copy of the prompt encoder and mask decoder,
plus a learned offset per structure. An update whose test IoU stays within the acceptance threshold of the best so far
is copied into the live copy that predictions use, and logged to the project's MLflow run as isonoai logs its models;
otherwise training is rolled back to the last accepted state.
"""

import json
import logging
import random
import tempfile
import threading
from pathlib import Path
from typing import Callable

import cloudpickle
import cv2
import keras
import mlflow
import numpy as np
import tensorflow as tf

import tsam.utils
from tsam.config import Config
from tsam.data import augment, build_augmentation, draw, mask_box, rasterize, read_test_set, structure_ids

logger = logging.getLogger(__name__)

MODEL = "model"  # the run's latest accepted SAM, logged as isonoai logs its models
CHECKPOINT = "checkpoint"  # what resuming needs besides it: structure offsets, replay buffer, step, best test IoU
EXAMPLES = 20  # test images drawn to examples/base (the base T-SAM) and examples/deployed (the latest accepted one)

# mlflow.tensorflow pickles Keras's registered classes along with the model. By value, so that isonoai, which has no
# tsam, can load it with mlflow.tensorflow.load_model as it loads its own models.
cloudpickle.register_pickle_by_value(tsam.utils)  # the SAM classes copied from isonoai


def mask_iou(logits: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
    """Per-pair IoU of the thresholded logits with the (B, 256, 256) masks; 1 when both are empty."""
    pred = tf.cast(logits > 0, tf.float32)
    inter = tf.reduce_sum(pred * masks, axis=[1, 2])
    union = tf.reduce_sum(tf.maximum(pred, masks), axis=[1, 2])
    return tf.where(union > 0, inter / tf.maximum(union, 1.0), 1.0)


def dice_loss(logits: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
    """1 - isonoai's dice coefficient (V-Net's soft Dice, smoothing 1), but per instance and averaged rather than over
    the whole batch, so a small instance counts as much as a large one, as in the test IoU."""
    prob = tf.sigmoid(logits)
    inter = tf.reduce_sum(masks * prob, axis=[1, 2])
    dice = (2 * inter + 1) / (tf.reduce_sum(masks**2, axis=[1, 2]) + tf.reduce_sum(prob**2, axis=[1, 2]) + 1)
    return 1 - tf.reduce_mean(dice)


# (key, name) in the config's loss_function -> loss of (mask logits, predicted IoU, masks)
LOSSES: dict[tuple[str, str], Callable] = {
    ("mask", "dice"): lambda logits, iou, masks: dice_loss(logits, masks),
    ("iou", "mse"): lambda logits, iou, masks: tf.reduce_mean((mask_iou(logits, masks) - iou) ** 2),
}


def clone(module: keras.Model) -> keras.Model:
    """A copy of a SAM module, with its own weights."""
    copy = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(module))
    copy.set_weights(module.get_weights())
    return copy


class Decoder:
    """T-SAM's prompt encoder and mask decoder, with a learned offset per structure added to the box's embeddings."""

    def __init__(self, prompt_encoder: keras.Model, mask_decoder: keras.Model, structures: int, conditioning: bool):
        self.prompt_encoder, self.mask_decoder = prompt_encoder, mask_decoder
        self.offsets = tf.Variable(tf.zeros((structures, prompt_encoder.dim))) if conditioning else None

    def __call__(self, embeddings, boxes, structures=None, training=False) -> tuple[tf.Tensor, tf.Tensor]:
        """Mask logits (B, 256, 256) and predicted IoU (B,) for (B, 4) boxes on (B, E, E, dim) embeddings.
        `structures` (B,) picks each box's offset; None for no offset."""
        prompts = self.prompt_encoder({"boxes": boxes}, training=training)
        if self.offsets is not None and structures is not None:
            prompts["prompt_sparse_embeddings"] += tf.gather(self.offsets, structures)[:, None]
        outputs = self.mask_decoder({"image_embeddings": embeddings, **prompts}, training=training)
        return outputs["mask"][:, 0], outputs["iou"][:, 0]

    @property
    def variables(self) -> list[tf.Variable]:
        offsets = [self.offsets] if self.offsets is not None else []
        return self.prompt_encoder.weights + self.mask_decoder.weights + offsets


class OnlineTrainer:
    """Trains T-SAM on one project's submitted labels, and keeps the live decoder that predictions use."""

    def __init__(self, config: Config, sam: keras.Model, load_image: Callable[[str], tf.Tensor]):
        """`sam` is the generic T-SAM, or with `continue_run_id`, that run's model. `load_image(prescan_key)` gives that
        prescan image as (256, 256, 1) float32 in [0, 1]."""
        self.config, self.sam, self.encoder, self.load_image = config, sam, sam.image_encoder, load_image
        self.structures = list(config.structures.mapping)
        self.label_structure = structure_ids(config.structures.mapping)
        n, conditioning = len(self.structures), config.structures.conditioning
        self.live = Decoder(sam.prompt_encoder, sam.mask_decoder, n, conditioning)
        self.train = Decoder(clone(sam.prompt_encoder), clone(sam.mask_decoder), n, conditioning)
        self.lock = threading.Lock()  # the live decoder is updated while predictions use it

        self.augmentation = build_augmentation(config.trainer.augmentation)
        self.losses = [(loss.weight, LOSSES[loss.key, loss.name]) for loss in config.trainer.loss_function]
        trained = [self.train.mask_decoder] + ([] if config.sam.prompt_encoder_frozen else [self.train.prompt_encoder])
        offsets = [self.train.offsets] if conditioning else []
        self.trainable = [w for module in trained for w in module.trainable_variables] + offsets
        opt = config.optimizer
        self.optimizer = keras.optimizers.get(
            {"class_name": opt.name, "config": {"learning_rate": opt.learning_rate, **opt.model_extra}}
        )
        self.optimizer.build(self.trainable)

        self.buffer: list[dict] = []  # every pair trained on so far (the replay buffer)
        self.queue: list[list[dict]] = []  # pairs of each submitted annotation waiting for the next update
        self.queue_lock = threading.Lock()
        self.updating = False
        self.step = 0  # updates so far: the step of every logged metric

        # The image encoder is frozen and test images aren't augmented, so their embeddings are computed once
        self.test_set = read_test_set(config.test.dataset_paths, self.label_structure)
        images = self.test_set["images"]  # kept for the examples
        self.test_set["embeddings"] = np.concatenate(
            [self.encoder(images[i : i + 16], training=False) for i in range(0, len(images), 16)]
        )

        self.mlflow = mlflow.MlflowClient()
        if config.continue_run_id:
            self.run_id = config.continue_run_id
            self.best = self._resume()
            self._log(self.test(self.train))
        else:
            experiment = mlflow.set_experiment(config.experiment_name)
            self.run_id = self.mlflow.create_run(experiment.experiment_id, run_name=config.run_name).info.run_id
            self.mlflow.log_dict(self.run_id, config.model_dump(mode="json", by_alias=True), "config.yaml")
            metrics = self.test(self.train)
            self.best = metrics["test_iou"]
            self._log(metrics)
            self._log_examples("base")
        self.accepted = self._state()
        logger.info(f"Online training into MLflow run {self.run_id}, from step {self.step}; test IoU {self.best:.4f}")

    def predict(self, embeddings: tf.Tensor, boxes: tf.Tensor, structure: int | None) -> tuple[np.ndarray, np.ndarray]:
        """The live decoder's mask logits and predicted IoU for boxes of one structure (None: no offset)."""
        structures = None if structure is None else tf.fill([len(boxes)], structure)
        with self.lock:
            logits, iou = self.live(embeddings, boxes, structures)
        return np.asarray(logits), np.asarray(iou)

    def add(self, pairs: list[dict]) -> None:
        """Queue one submitted annotation's pairs; each update takes the first `samples_per_update` queued."""
        if pairs:
            with self.queue_lock:
                self.queue.append(pairs)
            self._start_update()

    def _start_update(self) -> None:
        n = self.config.trainer.samples_per_update
        with self.queue_lock:
            if self.updating or len(self.queue) < n:
                return
            annotations, self.queue, self.updating = self.queue[:n], self.queue[n:], True
        threading.Thread(target=self._update, args=([p for a in annotations for p in a],), daemon=True).start()

    def _update(self, new: list[dict]) -> None:
        """`steps_per_update` gradient steps, each on new and replayed pairs drawn at random (with replacement, so the
        same number of steps however many pairs there are); then accept it, or roll it back."""
        try:
            t = self.config.trainer
            n_replay = round(t.minibatch_size * t.replay.ratio / (1 + t.replay.ratio)) if self.buffer else 0
            draw = lambda: random.choices(new, k=t.minibatch_size - n_replay) + random.choices(self.buffer, k=n_replay)
            losses = [self._train_step(draw()) for _ in range(t.steps_per_update)]
            self.buffer += new
            self.step += 1

            metrics = self.test(self.train)
            accepted = metrics["test_iou"] - self.best >= self.config.test.acceptance_threshold
            if accepted:
                with self.lock:
                    for live, train in zip(self.live.variables, self.train.variables):
                        live.assign(train)
                self.best = max(self.best, metrics["test_iou"])
                self.accepted = self._state()
                self._checkpoint()
            else:
                self._restore(self.accepted)
            self._log(
                metrics,
                train_loss=float(np.mean(losses)),
                accepted=float(accepted),
                new_pairs=len(new),
                buffer_pairs=len(self.buffer),
            )
            outcome = "accepted" if accepted else "rolled back"
            logger.info(f"Update {self.step}: test IoU {metrics['test_iou']:.4f}, {outcome}")
        except Exception:
            logger.exception("Online update failed")
        finally:
            with self.queue_lock:
                self.updating = False
        self._start_update()  # in case enough annotations arrived meanwhile

    def _train_step(self, pairs: list[dict]) -> float:
        """One gradient step on a minibatch of pairs, each freshly augmented."""
        image = {key: self.load_image(key) for key in {p["image"] for p in pairs}}
        images = tf.stack([image[p["image"]] for p in pairs])
        masks = tf.constant(np.stack([rasterize(p["polygons"]) for p in pairs])[..., None], tf.float32)
        images, masks = augment(self.augmentation, images, masks)
        keep = tf.reduce_sum(masks, axis=[1, 2, 3]) > 0  # augmentation can shift a mask out of the frame
        images, masks = images[keep], masks[keep][..., 0]
        structures = tf.constant([p["structure"] for p in pairs])[keep]
        boxes = mask_box(masks, self.config.trainer.box_jitter)

        embeddings = self.encoder(images, training=False)
        with tf.GradientTape() as tape:
            logits, iou = self.train(embeddings, boxes, structures, training=True)
            loss = sum(weight * fn(logits, iou, masks) for weight, fn in self.losses)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable), self.trainable))
        return float(loss)

    def test(self, decoder: Decoder) -> dict[str, float]:
        """Mean test IoU over every test pair (`test_iou`), and per structure (`test_iou/<structure>`)."""
        t, size = self.test_set, self.config.test.batch_size
        ious = []
        for i in range(0, len(t["image"]), size):
            masks = tf.constant(t["mask"][i : i + size], tf.float32)
            embeddings = t["embeddings"][t["image"][i : i + size]]
            logits, _ = decoder(embeddings, mask_box(masks), tf.constant(t["structure"][i : i + size]))
            ious.append(mask_iou(logits, masks).numpy())
        ious = np.concatenate(ious)
        metrics = {"test_iou": float(ious.mean())}
        for s, name in enumerate(self.structures):
            if np.any(t["structure"] == s):
                metrics[f"test_iou/{name}"] = float(ious[t["structure"] == s].mean())
        return metrics

    def _log(self, metrics: dict[str, float], **more: float) -> None:
        for key, value in {**metrics, **more}.items():
            self.mlflow.log_metric(self.run_id, key, value, step=self.step)

    def _state(self) -> list[np.ndarray]:
        """The training decoder's weights and the optimizer's state."""
        return [v.numpy() for v in self.train.variables + self.optimizer.variables]

    def _restore(self, state: list[np.ndarray]) -> None:
        for variable, value in zip(self.train.variables + self.optimizer.variables, state):
            variable.assign(value)

    def _checkpoint(self) -> None:
        """Log the accepted SAM (the live one) to the run, and what resuming needs besides it."""
        with tempfile.TemporaryDirectory() as folder:
            model, checkpoint = Path(folder) / MODEL, Path(folder) / CHECKPOINT
            mlflow.tensorflow.save_model(self.sam, str(model))
            self.mlflow.log_artifacts(self.run_id, str(model), MODEL)
            checkpoint.mkdir()
            if self.live.offsets is not None:
                offsets = dict(zip(self.structures, self.live.offsets.numpy().tolist()))
                (checkpoint / "offsets.json").write_text(json.dumps(offsets))
            (checkpoint / "buffer.json").write_text(json.dumps(self.buffer))
            (checkpoint / "progress.json").write_text(json.dumps({"step": self.step, "best": self.best}))
            self.mlflow.log_artifacts(self.run_id, str(checkpoint), CHECKPOINT)
        self._log_examples("deployed")

    def _log_examples(self, name: str) -> None:
        """Log the first EXAMPLES test images to examples/<name>: each its labels (left) beside the training
        decoder's masks from their boxes (right)."""
        t = self.test_set
        with tempfile.TemporaryDirectory() as folder:
            for image in np.unique(t["image"])[:EXAMPLES]:
                pairs = np.flatnonzero(t["image"] == image)
                masks, structures = t["mask"][pairs], t["structure"][pairs]
                boxes = mask_box(tf.constant(masks, tf.float32))
                logits, _ = self.train(t["embeddings"][t["image"][pairs]], boxes, tf.constant(structures))
                boxes, predicted = boxes.numpy(), logits.numpy() > 0
                panels = [draw(t["images"][image], m, boxes, structures, self.structures) for m in (masks, predicted)]
                cv2.imwrite(f"{folder}/{image:03d}.png", np.hstack(panels))
            self.mlflow.log_artifacts(self.run_id, folder, f"examples/{name}")

    def _resume(self) -> float:
        """Restore the run's checkpoint; the SAM is already its model (model.py). Returns its best test IoU.

        The optimizer starts afresh: its state isn't part of the logged model."""
        folder = Path(self.mlflow.download_artifacts(self.run_id, CHECKPOINT))
        if self.live.offsets is not None:
            offsets = json.loads((folder / "offsets.json").read_text())
            for decoder in (self.live, self.train):
                decoder.offsets.assign([offsets[name] for name in self.structures])
        self.buffer = json.loads((folder / "buffer.json").read_text())
        progress = json.loads((folder / "progress.json").read_text())
        self.step = progress["step"]
        return progress["best"]
