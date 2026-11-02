"""Online training of T-SAM on one Label Studio project, as its labels are submitted (see online-training.md).

The image encoder stays frozen, shared with serving. Training updates a copy of the prompt encoder and mask decoder.
An update whose test IoU stays within the acceptance threshold of the best so far
is copied into the live copy that predictions use, and logged to the project's MLflow run as isonoai logs its models;
otherwise training is rolled back to the last accepted state.
"""

import functools
import json
import logging
import random
import tempfile
import threading
import time
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
from tsam.data import augment, build_augmentation, draw, draw_accepted, mask_box, rasterize, read_test_set
from tsam.structures import SAM_STRUCTURES
from tsam.utils.sam import EXCLUDED_LOGIT

logger = logging.getLogger(__name__)

MODEL = "model"  # the run's latest accepted SAM, logged as isonoai logs its models
CHECKPOINT = "checkpoint"  # what resuming needs besides it: replay buffer, step, best test IoU
EXAMPLES = 20  # test images drawn to examples/base (the base T-SAM) and examples/deployed (the latest accepted one)

# mlflow.tensorflow pickles Keras's registered classes along with the model. By value, so that isonoai, which has no
# tsam, can load it with mlflow.tensorflow.load_model as it loads its own models.
cloudpickle.register_pickle_by_value(tsam.utils)  # the SAM classes copied from isonoai

# What MLflow would infer for the logged model, as isonoai's list; given, so it doesn't load the model in a subprocess
# to infer them (slow, and it fails here).
REQUIREMENTS = [f"tensorflow=={tf.__version__}", f"cloudpickle=={cloudpickle.__version__}"]
# isonoai's models have no signature either; MLflow warns about that at every save.
logging.getLogger("mlflow.tensorflow").setLevel(logging.ERROR)


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
    """T-SAM's prompt encoder and mask decoder: its whole pass after the image encoder."""

    def __init__(self, prompt_encoder: keras.Model, mask_decoder: keras.Model):
        self.prompt_encoder, self.mask_decoder = prompt_encoder, mask_decoder

    def __call__(self, embeddings, boxes, structures, accepted, training=False) -> tuple[tf.Tensor, tf.Tensor]:
        """Mask logits (B, 256, 256) and predicted IoU (B,) for (B, 4) boxes of (B,) structures on (B, E, E, dim)
        embeddings, given the (B, 256, 256) map of the accepted masks' structures; the logits are EXCLUDED_LOGIT on
        those masks, as in the SAM model's own pass."""
        prompts = {"boxes": boxes, "structures": structures, "accepted": accepted}
        prompts = self.prompt_encoder(prompts, training=training)
        excluded = prompts.pop("excluded")
        outputs = self.mask_decoder({"image_embeddings": embeddings, **prompts}, training=training)
        return tf.where(excluded, EXCLUDED_LOGIT, outputs["mask"][:, 0]), outputs["iou"][:, 0]

    def predict(
        self, embeddings: tf.Tensor, boxes: tf.Tensor, structure: int, accepted: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mask logits and predicted IoU for boxes of one structure (0: "Anything") on one image, given the
        (256, 256) map of its accepted masks' structures."""
        n = len(boxes)
        logits, iou = self(embeddings, boxes, tf.fill([n], structure), np.repeat(accepted[None], n, axis=0))
        return np.asarray(logits), np.asarray(iou)

    @property
    def variables(self) -> list[tf.Variable]:
        return self.prompt_encoder.weights + self.mask_decoder.weights


class OnlineTrainer:
    """Trains T-SAM on one project's submitted labels, and keeps the live decoder that predictions use."""

    def __init__(self, config: Config, sam: keras.Model, load_image: Callable[[str], tf.Tensor]):
        """`sam` is the generic T-SAM, or with `continue_run_id`, that run's model. `load_image(prescan_key)` gives that
        prescan image as (256, 256, 1) float32 in [0, 1]."""
        self.config, self.sam, self.encoder = config, sam, sam.image_encoder
        # Kept in memory (256 KB each): every step loads its minibatch's images, mostly ones loaded before, and S3 takes
        # ~0.1 s per image, many times the GPU's time for the step.
        self.load_image = functools.cache(load_image)
        self.structures = sam.prompt_encoder.structures
        if self.structures != list(SAM_STRUCTURES):  # the model's structure embeddings are by index
            raise ValueError(f"The model's structures, {self.structures}, aren't tsam/structures.py's")
        self.live = Decoder(sam.prompt_encoder, sam.mask_decoder)
        self.train = Decoder(clone(sam.prompt_encoder), clone(sam.mask_decoder))
        self.lock = threading.Lock()  # the live decoder is updated while predictions use it

        self.augmentation = build_augmentation(config.trainer.augmentation)
        self.losses = [(loss.weight, LOSSES[loss.key, loss.name]) for loss in config.trainer.loss_function]
        trained = [self.train.mask_decoder] + ([] if config.sam.prompt_encoder_frozen else [self.train.prompt_encoder])
        self.trainable = [w for module in trained for w in module.trainable_variables]
        opt = config.optimizer
        self.optimizer = keras.optimizers.get(
            {"class_name": opt.name, "config": {"learning_rate": opt.learning_rate, **opt.model_extra}}
        )
        self.optimizer.build(self.trainable)

        self.buffer: list[dict] = []  # every annotation trained on so far (the replay buffer), as data.py's records
        self.queue: list[dict] = []  # submitted annotations waiting for the next update
        self.rng = np.random.default_rng()  # draws the accepted masks
        self.queue_lock = threading.Lock()
        self.updating = False
        self.step = 0  # updates so far: the step of every logged metric

        # The image encoder is frozen and test images aren't augmented, so their embeddings are computed once
        self.test_set = read_test_set(config.test.dataset_paths)
        images = self.test_set["images"]  # kept for the examples
        logger.info(f"Test set: {len(self.test_set['image'])} instances on {len(images)} images; embedding them")
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

    def predict(
        self, embeddings: tf.Tensor, boxes: tf.Tensor, structure: int, accepted: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The live decoder's prediction (Decoder.predict)."""
        with self.lock:
            return self.live.predict(embeddings, boxes, structure, accepted)

    def add(self, annotation: dict) -> None:
        """Queue a submitted annotation (data.py's record); each update takes the first `samples_per_update` queued."""
        if annotation["instances"]:
            with self.queue_lock:
                self.queue.append(annotation)
            self._start_update()

    def _start_update(self) -> None:
        n = self.config.trainer.samples_per_update
        with self.queue_lock:
            if self.updating or len(self.queue) < n:
                return
            annotations, self.queue, self.updating = self.queue[:n], self.queue[n:], True
        threading.Thread(target=self._update, args=(annotations,), daemon=True).start()

    def _update(self, new: list[dict]) -> None:
        """`steps_per_update` gradient steps, each on new and replayed annotations drawn at random (with replacement,
        so the same number of steps however many there are); then accept it, or roll it back."""
        try:
            t, start, replay, best = self.config.trainer, time.perf_counter(), len(self.buffer), self.best
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
                new_annotations=len(new),
                buffer_annotations=len(self.buffer),
            )
            outcome, iou = "accepted" if accepted else "rolled back", metrics["test_iou"]
            logger.info(
                f"Update {self.step}: loss {np.mean(losses):.4f}, test IoU {iou:.4f} (best {best:.4f}), {outcome}, "
                f"in {time.perf_counter() - start:.0f}s ({len(new)} new annotations, {replay} in replay)"
            )
        except Exception:
            logger.exception("Online update failed")
        finally:
            with self.queue_lock:
                self.updating = False
        self._start_update()  # in case enough annotations arrived meanwhile

    def _train_step(self, annotations: list[dict]) -> float:
        """One gradient step on a pair drawn from each of a minibatch of annotations: a random instance, with a random
        share of the others accepted (draw_accepted); each freshly augmented, its accepted masks with it."""
        image = {key: self.load_image(key) for key in {a["image"] for a in annotations}}
        images = tf.stack([image[a["image"]] for a in annotations])
        one_hot = np.eye(len(self.structures), dtype=np.uint8)
        structures, layers = [], []  # each pair's structure, and its mask and accepted masks as channels
        for annotation in annotations:
            masks = np.stack([rasterize(instance["polygons"]) for instance in annotation["instances"]])
            ids = np.array([instance["structure"] for instance in annotation["instances"]])
            i = random.randrange(len(masks))
            # One-hot, so augmentation moves each structure's accepted pixels as it moves the mask
            accepted = one_hot[draw_accepted(masks, ids, i, self.rng)][..., 1:]
            structures.append(ids[i])
            layers.append(np.concatenate([masks[i, ..., None], accepted], axis=-1))
        images, layers = augment(self.augmentation, images, tf.constant(np.stack(layers), tf.float32))
        keep = tf.reduce_sum(layers[..., 0], axis=[1, 2]) > 0  # augmentation can shift a mask out of the frame
        images, layers, structures = images[keep], layers[keep], tf.constant(structures)[keep]
        masks = layers[..., 0]
        accepted = tf.cast(tf.reduce_max(layers[..., 1:] * tf.range(1.0, len(self.structures)), axis=-1), tf.int32)
        boxes = mask_box(masks, self.config.trainer.box_jitter)
        # As isonoai trains: against the best mask the model may give, the mask less the accepted ones
        target = masks * tf.cast(accepted == 0, tf.float32)

        embeddings = self.encoder(images, training=False)
        with tf.GradientTape() as tape:
            logits, iou = self.train(embeddings, boxes, structures, accepted, training=True)
            loss = sum(weight * fn(logits, iou, target) for weight, fn in self.losses)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable), self.trainable))
        return float(loss)

    def test(self, decoder: Decoder) -> dict[str, float]:
        """Mean test IoU over every test pair (`test_iou`), and per structure (`test_iou/<structure>`); against the
        whole mask, as isonoai's SAMTester scores."""
        t, size = self.test_set, self.config.test.batch_size
        ious = []
        for i in range(0, len(t["image"]), size):
            masks = tf.constant(t["mask"][i : i + size], tf.float32)
            embeddings = t["embeddings"][t["image"][i : i + size]]
            structures, accepted = t["structure"][i : i + size], t["accepted"][i : i + size]
            logits, _ = decoder(embeddings, mask_box(masks), structures, accepted)
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
            mlflow.tensorflow.save_model(self.sam, str(model), pip_requirements=REQUIREMENTS)
            self.mlflow.log_artifacts(self.run_id, str(model), MODEL)
            checkpoint.mkdir()
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
                logits, _ = self.train(t["embeddings"][t["image"][pairs]], boxes, structures, t["accepted"][pairs])
                boxes, predicted = boxes.numpy(), logits.numpy() > 0
                panels = [draw(t["images"][image], m, boxes, structures, self.structures) for m in (masks, predicted)]
                cv2.imwrite(f"{folder}/{image:03d}.png", np.hstack(panels))
            self.mlflow.log_artifacts(self.run_id, folder, f"examples/{name}")

    def _resume(self) -> float:
        """Restore the run's checkpoint; the SAM is already its model (model.py). Returns its best test IoU.

        The optimizer starts afresh: its state isn't part of the logged model."""
        folder = Path(self.mlflow.download_artifacts(self.run_id, CHECKPOINT))
        self.buffer = json.loads((folder / "buffer.json").read_text())
        progress = json.loads((folder / "progress.json").read_text())
        self.step = progress["step"]
        return progress["best"]
