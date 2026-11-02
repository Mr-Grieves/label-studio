"""The online-training config: a YAML file (e.g. configs/foot-ankle.yaml), validated with pydantic."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    """Unknown keys fail loudly instead of being silently dropped."""

    model_config = ConfigDict(extra="forbid")


class SAMConfig(Strict):
    base_run_id: str  # the MLflow run of the generic T-SAM to start from: one with structures (structures.py)
    prompt_encoder_frozen: bool = False


class OptimizerConfig(BaseModel):
    """Keys other than `name` go to the Keras optimizer as they are."""

    model_config = ConfigDict(extra="allow")

    name: str
    learning_rate: float


class ReplayConfig(Strict):
    ratio: float  # replayed annotations per new one in each minibatch


class LossConfig(Strict):
    key: Literal["mask", "iou"]
    name: Literal["dice", "mse"]
    weight: float = 1.0


class TrainerConfig(Strict):
    samples_per_update: int  # submitted annotations per update
    backfill_limit: int | None  # a new run first trains on the project's latest this many annotations; null: all
    minibatch_size: int  # pairs per gradient step, one drawn from each of as many annotations
    steps_per_update: int  # gradient steps per update, each on a minibatch drawn from its new and replayed annotations
    box_jitter: float
    replay: ReplayConfig
    augmentation: list[dict[str, dict[str, Any]]]  # keras.layers class name -> its arguments
    loss_function: list[LossConfig]


class TestConfig(Strict):
    dataset_paths: list[Path]  # relative paths are under $DATA_DIR (default /data)
    batch_size: int
    acceptance_threshold: float  # the most an update may drop the test IoU below the best so far


# Needed only to train: a config without trainer_config serves the model frozen (e.g. configs/frozen.yaml)
TRAINING = ("project_id", "experiment_name", "run_name", "optimizer", "test")


class Config(Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    description: str
    version: str
    continue_run_id: str | None = None  # resume this run; when frozen, serve its model
    sam: SAMConfig = Field(alias="model_config")  # `model_config` is pydantic's own name
    trainer: TrainerConfig | None = Field(None, alias="trainer_config")
    project_id: int | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    optimizer: OptimizerConfig | None = Field(None, alias="optimizer_config")
    test: TestConfig | None = Field(None, alias="test_config")

    @model_validator(mode="after")
    def training_is_complete(self) -> "Config":
        missing = [name for name in TRAINING if getattr(self, name) is None]
        if self.trainer and missing:
            raise ValueError(f"Training (trainer_config) also needs {', '.join(missing)}")
        return self


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f))
