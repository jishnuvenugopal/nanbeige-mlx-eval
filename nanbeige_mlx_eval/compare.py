"""Diff two persisted runs.

Like :mod:`report`, comparison reads only persisted artifacts. It highlights
per-case pass flips and latency deltas between two runs (for example the same
suite across two quantization levels).
"""

from __future__ import annotations

import json
from pathlib import Path

from .report import _read_artifacts


def write_compare(run_a: str | Path, run_b: str | Path, output: str | Path | None = None) -> Path:
    a = _read_artifacts(run_a)
    b = _read_artifacts(run_b)
    md = _render_compare(a, b)
    out = Path(output) if output else a["dir"] / f"compare_{b['manifest']['run_id']}.md"
    out.write_text(md, encoding="utf-8")
    return out


def _render_compare(a: dict, b: dict) -> str:
    ma, mb = a["manifest"], b["manifest"]
    sa, sb = a["summary"], b["summary"]
    lines: list[str] = []
    lines.append("# Comparison\n")
    lines.append(f"- A: `{ma.get('model') or ma['runtime']}` → {sa['pass_rate'] * 100:.1f}%\n")
    lines.append(f"- B: `{mb.get('model') or mb['runtime']}` → {sb['pass_rate'] * 100:.1f}%\n")
    delta = (sb["pass_rate"] - sa["pass_rate"]) * 100
    lines.append(f"- pass-rate delta (B − A): **{delta:+.1f} points**\n")

    la, lb = sa.get("latency", {}), sb.get("latency", {})
    if any(la.values()) or any(lb.values()):
        lines.append("\n## Latency delta (B − A)\n")
        lines.append("| metric | A | B | Δ |\n|---|---|---|---|\n")
        for key, unit in (
            ("decode_tps_aggregate", " tok/s"),
            ("decode_tps_median", " tok/s"),
            ("ttft_s_median", " s"),
            ("peak_rss_mb", " MB"),
        ):
            va, vb = la.get(key, 0), lb.get(key, 0)
            lines.append(f"| {key} | {va}{unit} | {vb}{unit} | {vb - va:+.2f}{unit} |\n")

    ra = {r["id"]: r for r in a["results"]}
    rb = {r["id"]: r for r in b["results"]}
    flips = []
    for cid in sorted(set(ra) & set(rb)):
        pa, pb = ra[cid]["pass"], rb[cid]["pass"]
        if pa != pb:
            flips.append((cid, pa, pb))
    lines.append("\n## Per-case pass flips\n")
    if not flips:
        lines.append("_none_\n")
    else:
        lines.append("| id | A | B |\n|---|---|---|\n")
        for cid, pa, pb in flips:
            lines.append(f"| {cid} | {'✅' if pa else '❌'} | {'✅' if pb else '❌'} |\n")
    return "".join(lines)
