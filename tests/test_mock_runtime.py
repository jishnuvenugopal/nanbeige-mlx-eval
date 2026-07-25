"""The mock runtime must pass every case of the smoke suite.

This is the harness-readiness guarantee: a fully-passing mock run proves the
plumbing (grading, artifact writing, idempotent reporting) is correct, with no
model download required.
"""

from pathlib import Path

from nanbeige_mlx_eval.report import _read_artifacts, write_report
from nanbeige_mlx_eval.runtime import MockRuntime, run_suite

SUITES_DIR = Path(__file__).resolve().parent.parent / "suites"


def test_smoke_suite_passes_on_mock(tmp_path):
    run_dir = run_suite(SUITES_DIR / "smoke.json", MockRuntime(), tmp_path, run_id="test")
    art = _read_artifacts(run_dir)
    assert art["summary"]["n_pass"] == art["summary"]["n_cases"]
    assert art["summary"]["pass_rate"] == 1.0


def test_report_is_idempotent(tmp_path):
    run_dir = run_suite(SUITES_DIR / "smoke.json", MockRuntime(), tmp_path, run_id="t2")
    p1 = write_report(run_dir)
    p2 = write_report(run_dir)
    assert p1 == p2
    assert p1.exists()
    assert "Run report" in p1.read_text()


def test_smoke_flag_when_limited(tmp_path):
    run_dir = run_suite(SUITES_DIR / "smoke.json", MockRuntime(), tmp_path, limit=2, run_id="t3")
    art = _read_artifacts(run_dir)
    assert art["manifest"]["settings"]["smoke"] is True
    assert "smoke run" in write_report(run_dir).read_text()
