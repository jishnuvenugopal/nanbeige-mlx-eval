"""Artifact-driven report regeneration.

Reports are regenerated solely from persisted artifacts (manifest / results /
summary), never from live computation. This makes ``report`` idempotent and
safe to re-run: the same run directory always yields the same report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_artifacts(run_dir: str | Path) -> dict[str, Any]:
    d = Path(run_dir)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    results = [
        json.loads(line)
        for line in (d / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {"manifest": manifest, "summary": summary, "results": results, "dir": d}


def write_report(run_dir: str | Path) -> Path:
    """Write (or overwrite) ``report.md`` in ``run_dir`` from its artifacts."""
    art = _read_artifacts(run_dir)
    d = art["dir"]
    m = art["manifest"]
    s = art["summary"]
    md = _render(art)
    (d / "report.md").write_text(md, encoding="utf-8")
    return d / "report.md"


def _render(art: dict[str, Any]) -> str:
    m, s = art["manifest"], art["summary"]
    lines: list[str] = []
    lines.append(f"# Run report — {m['suite']} ({m['runtime']})\n")
    lines.append(f"- **run_id:** `{m['run_id']}`\n")
    lines.append(f"- **model:** `{m.get('model') or 'n/a (mock)'}`\n")
    env = m.get("env", {})
    lines.append(
        f"- **env:** python {env.get('python')} · mlx {env.get('mlx_version')} · {env.get('machine')}\n"
    )
    if m.get("settings", {}).get("smoke"):
        lines.append(
            "> ⚠️ **smoke run** — case count was limited, so this is a "
            "harness-readiness check, not a benchmark-quality result.\n"
        )
    lines.append("\n## Summary\n")
    lines.append(
        f"- pass rate: **{s['n_pass']}/{s['n_cases']}** "
        f"({s['pass_rate'] * 100:.1f}%)\n"
    )
    lines.append("\n| grade kind | pass / n |\n|---|---|\n")
    for kind, v in sorted(s.get("by_kind", {}).items()):
        lines.append(f"| {kind} | {v['pass']} / {v['n']} |\n")
    lat = s.get("latency", {})
    if any(lat.values()):
        lines.append("\n## Latency / memory (real-model cases only)\n")
        lines.append("| metric | value |\n|---|---|\n")
        lines.append(f"| mean tokens / second | {lat.get('mean_tps', 0)} |\n")
        lines.append(f"| mean time-to-first-token | {lat.get('mean_ttft_s', 0)} s |\n")
        lines.append(f"| mean total time | {lat.get('mean_total_s', 0)} s |\n")
        lines.append(f"| mean generated tokens | {lat.get('mean_gen_tokens', 0)} |\n")
        lines.append(f"| peak resident memory | {lat.get('peak_rss_mb', 0)} MB |\n")
    lines.append("\n## Cases\n")
    lines.append("| id | kind | pass | detail | tok/s | tokens |\n|---|---|---|---|---|---|\n")
    for r in art["results"]:
        lines.append(
            f"| {r['id']} | {r['grade_kind']} | {'✅' if r['pass'] else '❌'} | "
            f"{r['detail']} | {r.get('tps', 0)} | {r.get('generated_tokens', 0)} |\n"
        )
    return "".join(lines)
