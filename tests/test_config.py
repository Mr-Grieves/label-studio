"""config.py, and the project configs in tsam/configs."""

from pathlib import Path

import pytest

from tsam.config import Config, load_config

CONFIGS = sorted((Path(__file__).parents[1] / "tsam" / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_project_config_loads(path):
    config = load_config(path)
    assert config.trainer is None or config.trainer.steps_per_update > 0 and config.trainer.samples_per_update > 0


def test_training_needs_its_sections(config):
    with pytest.raises(ValueError, match="test"):
        Config.model_validate({**config.model_dump(by_alias=True), "test_config": None})
