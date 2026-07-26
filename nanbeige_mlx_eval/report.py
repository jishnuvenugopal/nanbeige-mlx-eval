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
    mlx_v = env.get("mlx") or env.get("mlx_version") or "unknown"
    lines.append(
        f"- **env:** python {env.get('python')} · mlx {mlx_v} · mlx-lm {env.get('mlx_lm', '?')} "
        f"· {env.get('machine')} · git {env.get('git_commit', '?')}\n"
    )
    if m.get("quantization"):
        q = m["quantization"]
        lines.append(f"- **quantization:** {q.get('bits')}-bit, group_size={q.get('group_size')}\n")
    settings = m.get("settings", {})
    if settings.get("smoke"):
        lines.append(
            "> ⚠️ **smoke run** — case count was limited, so this is a "
            "harness-readiness check, not a benchmark-quality result.\n"
        )
    lines.append("\n## Summary\n")
    ci = s.get("pass_rate_ci95")
    ci_str = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
    lines.append(
        f"- pass rate: **{s['n_pass']}/{s['n_cases']}** "
        f"({s['pass_rate'] * 100:.1f}%){ci_str}\n"
    )
    trunc = s.get("truncated") or {}
    if trunc.get("n"):
        lines.append(
            f"- ⚠️ **{trunc['n']} case(s) truncated** (hit token cap, no eos): "
            f"{', '.join(trunc.get('ids', []))}\n"
        )
    lines.append("\n| grade kind | pass / n |\n|---|---|\n")
    for kind, v in sorted(s.get("by_kind", {}).items()):
        lines.append(f"| {kind} | {v['pass']} / {v['n']} |\n")
    lat = s.get("latency", {}) or {}
    if any(lat.values()):
        lines.append("\n## Latency / memory (real-model cases only)\n")
        lines.append("| metric | value |\n|---|---|\n")
        lines.append(f"| decode throughput (aggregate) | {lat.get('decode_tps_aggregate', 0)} tok/s |\n")
        lines.append(f"| decode throughput (median) | {lat.get('decode_tps_median', 0)} tok/s |\n")
        regime = lat.get("ttft_s_by_regime") or {}
        wt, bp = regime.get("with_tools"), regime.get("bare_prompt")
        lines.append(f"| TTFT median | {lat.get('ttft_s_median', 0)} s |\n")
        lines.append(f"| TTFT with tools / bare prompt | {wt} / {bp} s |\n")
        lines.append(f"| mean generated tokens | {lat.get('mean_gen_tokens', 0)} |\n")
        lines.append(f"| peak allocator memory | {lat.get('peak_rss_mb', 0)} MB |\n")
    lines.append("\n## Cases\n")
    lines.append("| id | kind | pass | detail | stop | tok/s | tokens |\n|---|---|---|---|---|---|---|\n")
    for r in art["results"]:
        stop = (r.get("stop_reason") or "—")
        lines.append(
            f"| {r['id']} | {r['grade_kind']} | {'✅' if r['pass'] else '❌'} | "
            f"{r['detail']} | {stop} | {r.get('tps', 0)} | {r.get('generated_tokens', 0)} |\n"
        )
    return "".join(lines)
