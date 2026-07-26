"""Retroactive re-grading of a persisted run.

Grading changes shouldn't force a model re-run. This module re-grades the
``output`` field already written to a run's ``results.jsonl`` against the suite,
applies the current graders (including reasoning isolation), and rewrites
``results.jsonl`` + ``summary.json`` + ``report.md`` in place.

Latency/memory/timing fields are preserved unchanged (they describe the
generation, which is unaffected by grader changes). Only ``pass``, ``detail``
and (optionally) ``stop_reason``-derived summaries move.

This is permanent infrastructure: you will change the grader again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .grading import grade
from .report import write_report
from .runtime import _write_summary
from .suite import load_suite


def _read_results(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _read_summary_suite_name(run_dir: Path) -> str:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["suite"]


def regrade(run_dir: str | Path, suite_path: str | Path, *, require_answer: bool = True) -> Path:
    """Re-grade a persisted run against ``suite_path``; rewrite its artifacts."""
    d = Path(run_dir)
    if not (d / "results.jsonl").exists():
        raise FileNotFoundError(f"no results.jsonl in {d}")

    suite = load_suite(suite_path)
    expects = {c["id"]: c["expect"] for c in suite["cases"]}

    results = _read_results(d)
    n_changed = 0
    for r in results:
        expect = expects.get(r["id"])
        if expect is None:
            continue
        verdict = grade(expect, r["output"], require_answer=require_answer)
        if verdict["pass"] != r["pass"] or verdict["detail"] != r["detail"]:
            n_changed += 1
        r["pass"] = verdict["pass"]
        r["detail"] = verdict["detail"]
        r["grade_kind"] = expect["type"]

    # Rewrite results.jsonl (output + timing preserved; pass/detail updated).
    with (d / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    suite_name = _read_summary_suite_name(d)
    is_smoke = json.loads((d / "summary.json").read_text(encoding="utf-8")).get("smoke", False)
    _write_summary(d, results, suite_name, is_smoke)
    write_report(d)

    print(f"regraded {len(results)} cases ({n_changed} verdict(s) changed) in {d}")
    return d
