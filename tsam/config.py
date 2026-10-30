"""The online-training config: a YAML file (e.g. configs/foot-ankle.yaml), validated with pydantic."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    """Unknown keys fail loudly instead of being silently dropped."""

    model_config = ConfigDict(extra="forbid")


class StructuresConfig(Strict):
    conditioning: bool  # a learned offset per structure, added to its boxes' prompt embeddings
    mapping: dict[str, list[str]]  # structure -> its label names, without their tool suffix ("ATFL [SAM]" -> ATFL)


class SAMConfig(Strict):
    base_run_id: str  # the MLflow run of the generic T-SAM to start from
    prompt_encoder_frozen: bool = False


class OptimizerConfig(BaseModel):
    """Keys other than `name` go to the Keras optimizer as they are."""

    model_config = ConfigDict(extra="allow")

    name: str
    learning_rate: float


class ReplayConfig(Strict):
    ratio: float  # replayed pairs per new pair


class LossConfig(Strict):
    key: Literal["mask", "iou"]
    name: Literal["dice", "mse"]
    weight: float = 1.0


class TrainerConfig(Strict):
    samples_per_update: int  # submitted annotations per update
    backfill_limit: int | None  # a new run first trains on the project's latest this many annotations; null: all
    minibatch_size: int  # image+label pairs per gradient step
    box_jitter: float
    replay: ReplayConfig
    augmentation: list[dict[str, dict[str, Any]]]  # keras.layers class name -> its arguments
    loss_function: list[LossConfig]


class TestConfig(Strict):
    dataset_paths: list[Path]  # relative paths are under $DATA_DIR (default /data)
    batch_size: int
    acceptance_threshold: float  # the most an update may drop the test IoU below the best so far


class Config(Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    project_id: int
    description: str
    experiment_name: str
    run_name: str
    version: str
    continue_run_id: str | None = None
    structures: StructuresConfig
    sam: SAMConfig = Field(alias="model_config")  # `model_config` is pydantic's own name
    optimizer: OptimizerConfig = Field(alias="optimizer_config")
    trainer: TrainerConfig = Field(alias="trainer_config")
    test: TestConfig = Field(alias="test_config")


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f))
