"""AnyGoal Cube scoring in scripts/anygoal/eval_lewam_anygoal_cube.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

LW = Path(__file__).resolve().parents[1]
TOY_CONFIGS = LW.parents[1] / "external" / "AnyGoal" / "anygoal" / "resources" / "plan" / "toy_predictors"


def _script():
    spec = importlib.util.spec_from_file_location("eval_anygoal_cube", LW / "scripts" / "anygoal" / "eval_lewam_anygoal_cube.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_criterion_copied_into_success():
    metrics = {"success_rate": 0.0, "episode_successes": np.zeros(3, dtype=bool),
               "cube_hop_criterion_3": {"success_rate": 66.7, "episode_successes": [True, False, True]}}
    out = _script().anygoal_cube_success(metrics, "cube_hop_criterion_3")
    assert out["success_rate"] == 66.7
    np.testing.assert_array_equal(out["episode_successes"], [True, False, True])


def test_missing_criterion_raises():
    with pytest.raises(AssertionError, match="cube_hop_criterion_3"):
        _script().anygoal_cube_success({"success_rate": 0.0}, "cube_hop_criterion_3")


@pytest.mark.parametrize("env_name,config", [("anygoal/CubeHop-v0", "cube_hop_from_cube.yaml"),
                                              ("anygoal/CubeDiagonal-v0", "cube_diagonal.yaml")])
def test_success_key_matches_anygoal_config(env_name, config):
    criterion = yaml.safe_load((TOY_CONFIGS / config).read_text())["output"]["success_criterion"]
    assert criterion == "criterion_3"
    assert _script().SUCCESS_KEY[env_name] == "cube_hop_criterion_3"
