"""The manifest's reproducibility fields must carry real values, not 'unknown'.

B5 in CODE_REVIEW.md: every committed manifest had ``mlx_version: "unknown"``
because the code read ``mlx.__version__`` (which doesn't exist) instead of
``mlx.core.__version__``. This test guards against that rotting again.
"""

import sys

import pytest


def test_env_info_has_real_mlx_version():
    from nanbeige_mlx_eval.runtime import _env_info

    env = _env_info()
    # mlx.core.__version__ exists and is a version-like string.
    assert "mlx" in env, env
    assert env["mlx"] != "unknown", f"mlx version is 'unknown': {env}"
    assert isinstance(env["mlx"], str) and env["mlx"]
    # mlx-lm and transformers presence is recorded (they may be 'absent' on a
    # stripped CI image, but the key must exist).
    for key in ("mlx_lm", "mlx_nanbeige", "transformers"):
        assert key in env
    assert "git_commit" in env


def test_manifest_env_round_trip(tmp_path):
    # A real run writes the env block into manifest.json; smoke-check the shape.
    from nanbeige_mlx_eval.runtime import _write_manifest

    suite = {"suite": "x", "language": "en", "category": "test", "cases": []}
    _write_manifest(
        tmp_path, suite, type("R", (), {"name": "mock", "model_path": None})(),
        "t", 0, False, None, warmup=0, repeats=1,
    )
    import json
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["env"]["mlx"] != "unknown"
    assert m["settings"]["warmup"] == 0
    assert m["settings"]["repeats"] == 1


def test_wilson_interval_bounds():
    from nanbeige_mlx_eval.profile import wilson

    lo, hi = wilson(8, 8)
    # 8/8 -> point estimate 1.0; Wilson pulls the lower bound below 1.
    assert lo < 1.0 <= hi
    lo5, hi5 = wilson(4, 8)
    assert 0.0 <= lo5 < 0.5 < hi5 <= 1.0
    assert wilson(0, 0) == (0.0, 0.0)


def test_mlx_peak_mb_runs():
    # Smoke: mlx_peak_mb returns a float and doesn't raise on this build.
    from nanbeige_mlx_eval.profile import mlx_peak_mb, reset_peak_memory

    reset_peak_memory()
    val = mlx_peak_mb()
    assert isinstance(val, float)
    assert val >= 0.0
