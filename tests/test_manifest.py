"""The manifest's reproducibility fields must carry real values, not 'unknown'.

B5 in docs/investigation-log.md: every committed manifest had ``mlx_version: "unknown"``
because the code read ``mlx.__version__`` (which doesn't exist) instead of
``mlx.core.__version__``. This test guards against that rotting again.
"""

import sys

import pytest


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except Exception:
        return False


def test_env_info_has_real_mlx_version():
    from nanbeige_mlx_eval.runtime import _env_info

    env = _env_info()
    # mlx.core.__version__ exists and is a version-like string when mlx is
    # importable. On a runner without mlx (no linux x86_64 wheel), _env_info
    # stamps "unavailable" rather than raising -- that's the correct behaviour
    # for a model-free path, so accept it when mlx genuinely isn't installed.
    assert "mlx" in env, env
    if _mlx_available():
        assert env["mlx"] not in ("unknown", "unavailable"), f"mlx present but version not real: {env}"
        assert isinstance(env["mlx"], str) and env["mlx"]
    else:
        assert env["mlx"] == "unavailable", f"mlx absent but env says {env['mlx']!r}"
    # mlx-lm and transformers presence is recorded (they may be 'absent' on a
    # stripped CI image, but the key must exist).
    for key in ("mlx_lm", "nanbeige_mlx", "transformers"):
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
    # Accept the real version when mlx is present, "unavailable" when it isn't.
    if _mlx_available():
        assert m["env"]["mlx"] not in ("unknown", "unavailable")
    else:
        assert m["env"]["mlx"] == "unavailable"
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


@pytest.mark.skipif(not _mlx_available(), reason=(
    "mlx_peak_mb measures Metal memory; mlx not importable here"))
def test_mlx_peak_mb_runs():
    # Smoke: mlx_peak_mb returns a float and doesn't raise on this build.
    from nanbeige_mlx_eval.profile import mlx_peak_mb, reset_peak_memory

    reset_peak_memory()
    val = mlx_peak_mb()
    assert isinstance(val, float)
    assert val >= 0.0
