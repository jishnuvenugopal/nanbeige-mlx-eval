"""Regrade retroactively applies the current graders to persisted outputs.

The motivating case (B1 in docs/investigation-log.md): the committed 8-bit EN
`en-json-profile` run graded `schema_valid` on output that was reasoning
truncated mid-sentence. After the grader fix, regrade must flip it to a fail
without re-running the model.
"""

from __future__ import annotations

import json
from pathlib import Path

from nanbeige_mlx_eval.regrade import regrade
from nanbeige_mlx_eval.runtime import _write_manifest, _write_results, _write_summary

SUITE_PATH = Path(__file__).resolve().parent.parent / "suites" / "agentic_en.json"


def _write_run(run_dir: Path, rows: list[dict], *, smoke: bool = False) -> None:
    """Write minimal artifacts shaped like a real run, with stale verdicts."""
    run_dir.mkdir(parents=True, exist_ok=True)
    suite = {"suite": "agentic_en", "language": "en", "category": "tool_use", "cases": []}
    _write_manifest(run_dir, suite, type("R", (), {"name": "mlx", "model_path": "x"})(),
                    "test", len(rows), smoke, None, warmup=0, repeats=1)
    _write_results(run_dir, rows)
    _write_summary(run_dir, rows, "agentic_en", smoke)


def test_regrade_flips_truncated_json_false_positive(tmp_path):
    # The committed 8-bit EN output: reasoning that never closed <think> nor
    # emitted an answer, but happened to contain a conforming JSON draft. Stored
    # with the OLD (wrong) verdict pass/schema_valid to simulate a stale run.
    truncated_output = (
        "The user wants a profile JSON. A profile has name and age. "
        'Typically we output it cleanly.\n\n Let me write it'
    )
    rows = [{
        "id": "en-json-profile",
        "pass": True,                       # the stale false-positive verdict
        "detail": "schema_valid",
        "grade_kind": "json_schema",
        "stop_reason": "length",
        "output_hash": "deadbeef",
        "output": truncated_output,
        "prompt_tokens": 57,
        "generated_tokens": 384,
        "ttft_s": 0.6,
        "total_s": 17.7,
        "tps": 21.6,
        "peak_rss_mb": 3167.0,
    }]
    run_dir = tmp_path / "run"
    _write_run(run_dir, rows)

    regrade(run_dir, SUITE_PATH, require_answer=True)

    new_rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    assert new_rows[0]["pass"] is False
    # The reasoning block had no <think> tag in this particular truncation, so
    # the whole stream is graded as the answer and fails for lack of a JSON
    # object. (An open <think> with no close would fail as truncated_no_answer.)
    assert new_rows[0]["detail"] in ("truncated_no_answer", "no_json_object_found")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n_pass"] == 0


def test_regrade_preserves_timing(tmp_path):
    # Latency/memory must survive a regrade (they describe the generation, which
    # we are not re-running).
    good_output = '{"name": "Alex", "age": 30}'
    rows = [{
        "id": "en-json-profile",
        "pass": False,
        "detail": "stale_wrong_verdict",
        "grade_kind": "json_schema",
        "stop_reason": "stop",
        "output_hash": "abc",
        "output": good_output,
        "prompt_tokens": 57,
        "generated_tokens": 50,
        "ttft_s": 0.5,
        "total_s": 3.0,
        "tps": 16.6,
        "peak_rss_mb": 3000.0,
    }]
    run_dir = tmp_path / "run"
    _write_run(run_dir, rows)

    regrade(run_dir, SUITE_PATH, require_answer=True)

    new = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()][0]
    assert new["pass"] is True
    assert new["detail"] == "schema_valid"
    # Timing fields preserved verbatim.
    for k in ("prompt_tokens", "generated_tokens", "ttft_s", "total_s", "tps", "peak_rss_mb"):
        assert new[k] == rows[0][k]
