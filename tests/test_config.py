"""config.py, and the project configs in tsam/configs."""

from pathlib import Path

import pytest

from tsam.config import load_config

CONFIGS = sorted((Path(__file__).parents[1] / "tsam" / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_project_config_loads(path):
    config = load_config(path)
    names = [name for names in config.structures.mapping.values() for name in names]
    assert len(names) == len(set(names)), "a label name maps to two structures"
    assert config.trainer.steps_per_update > 0 and config.trainer.samples_per_update > 0
