"""Run-times and the run orchestrator.

Two run-times share a common ``run_case`` interface:

* :class:`MockRuntime` – the default. Deterministic, requires no model. It
  returns a canned, expectation-shaped response so the full pipeline (grading,
  artifact writing, reporting) can be exercised in CI or triage without a
  download. It never makes a capability *claim* about the real model.

* :class:`MLXRuntime` – the real thing. Loads a converted MLX repo and generates
  with greedy decoding for reproducibility. Gated behind an explicit model path
  (there is no ambient "download on demand": you must point at a repo you
  converted).

The orchestrator :func:`run_suite` executes a suite over a run-time and writes
the standard four artifacts (manifest / results / summary / report). Runs are
**smoke-flagged** when ``limit`` caps the case count, which disables any
benchmark-quality framing in downstream reports.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import grading, profile
from .suite import SuiteError, load_suite, validate_suite


@dataclass
class CaseResult:
    output: str
    profile: Optional[profile.GenProfile] = None
    meta: dict[str, Any] = field(default_factory=dict)


class MockRuntime:
    """Deterministic, model-free run-time for harness plumbing."""

    name = "mock"

    def __init__(self, *_args, **_kwargs):
        pass

    def run_case(self, case: dict[str, Any]) -> CaseResult:
        expect = case.get("expect", {})
        kind = expect.get("type")
        if kind == "tool_call":
            payload = {"name": expect.get("tool"), "arguments": expect.get("args", {})}
            out = "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"
        elif kind == "choice":
            out = str((expect.get("options") or [""])[0])
        elif kind == "json_schema":
            # Synthesize a value that matches the schema's property types so the
            # json_schema grader is exercised on the mock path.
            props = (expect.get("schema") or {}).get("properties") or {}
            obj: dict[str, Any] = {}
            for k, spec in props.items():
                t = (spec or {}).get("type")
                obj[k] = (
                    "mock" if t == "string"
                    else 1 if t in ("number", "integer")
                    else True if t == "boolean"
                    else [] if t == "array"
                    else {} if t == "object"
                    else "mock"
                )
            out = json.dumps(obj, ensure_ascii=False)
        elif kind in ("exact_match", "contains"):
            out = str(expect.get("value", ""))
        else:
            out = "mock-output"
        return CaseResult(output=out, meta={"runtime": "mock"})


class MLXRuntime:
    """Real MLX run-time. Greedy decoding for reproducible grading."""

    name = "mlx"

    def __init__(self, model_path: str, default_max_tokens: int = 768):
        # Imported lazily so the harness imports without mlx_lm at collection time.
        from mlx_lm import load, stream_generate  # noqa: F401

        self._stream_generate = stream_generate
        self.model, self.tokenizer = load(model_path)
        self.model_path = model_path
        self.default_max_tokens = default_max_tokens

    def _build_prompt(self, case: dict[str, Any]) -> str:
        messages: list[dict[str, Any]] = []
        if case.get("context"):
            messages.append({"role": "system", "content": case["context"]})
        messages.append({"role": "user", "content": case["prompt"]})
        tools = case.get("tools")
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def run_case(self, case: dict[str, Any]) -> CaseResult:
        prompt = self._build_prompt(case)
        max_tokens = case.get("max_tokens", self.default_max_tokens)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        text_parts: list[str] = []
        n = 0
        ttft: Optional[float] = None
        t0 = profile.now()
        for resp in self._stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens):
            if n == 0:
                ttft = profile.now() - t0
            text_parts.append(resp.text)
            n += 1
        total = profile.now() - t0

        gen = profile.GenProfile(
            prompt_tokens=len(prompt_ids),
            generated_tokens=n,
            total_seconds=total,
            ttft_seconds=ttft or 0.0,
            peak_rss_mb=profile.peak_rss_mb(),
        )
        return CaseResult(
            output="".join(text_parts),
            profile=gen,
            meta={"runtime": "mlx", "model": str(self.model_path), "greedy": True},
        )


def _env_info() -> dict[str, Any]:
    try:
        import mlx

        mlx_version = getattr(mlx, "__version__", "unknown")
    except Exception:
        mlx_version = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx_version": mlx_version,
    }


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def run_suite(
    suite_path: str | Path,
    runtime,
    output_root: str | Path,
    *,
    limit: Optional[int] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Execute ``suite_path`` over ``runtime``; write artifacts under ``output_root``."""
    suite = load_suite(suite_path)
    validate_suite(suite)
    suite_name = suite["suite"]

    cases = suite["cases"]
    is_smoke = limit is not None and limit < len(cases)
    if limit is not None:
        cases = cases[:limit]

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / f"{suite_name}-{runtime.name}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for case in cases:
        cr = runtime.run_case(case)
        verdict = grading.grade(case["expect"], cr.output)
        results.append(
            {
                "id": case["id"],
                "pass": verdict["pass"],
                "detail": verdict["detail"],
                "grade_kind": case["expect"]["type"],
                "output_hash": _hash_text(cr.output),
                "output": cr.output,
                "prompt_tokens": cr.profile.prompt_tokens if cr.profile else 0,
                "generated_tokens": cr.profile.generated_tokens if cr.profile else 0,
                "ttft_s": round(cr.profile.ttft_seconds, 4) if cr.profile else 0.0,
                "total_s": round(cr.profile.total_seconds, 4) if cr.profile else 0.0,
                "tps": round(cr.profile.tokens_per_second, 2) if cr.profile else 0.0,
                "peak_rss_mb": round(cr.profile.peak_rss_mb, 1) if cr.profile else 0.0,
            }
        )

    _write_manifest(run_dir, suite, runtime, run_id, len(cases), is_smoke, limit)
    _write_results(run_dir, results)
    _write_summary(run_dir, results, suite_name, is_smoke)

    from .report import write_report

    write_report(run_dir)
    return run_dir


def _write_manifest(run_dir, suite, runtime, run_id, n_cases, is_smoke, limit):
    manifest = {
        "run_id": run_id,
        "suite": suite["suite"],
        "suite_language": suite.get("language"),
        "suite_category": suite.get("category"),
        "runtime": runtime.name,
        "model": getattr(runtime, "model_path", None),
        "env": _env_info(),
        "settings": {
            "decoding": "greedy" if runtime.name == "mlx" else "n/a",
            "default_max_tokens": getattr(runtime, "default_max_tokens", None),
            "limit": limit,
            "smoke": is_smoke,
        },
        "n_cases": n_cases,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_results(run_dir, results):
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            # Answer-key protection: only the grade outcome and model output are
            # persisted, never the suite's expected answer.
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_summary(run_dir, results, suite_name, is_smoke):
    n = len(results)
    n_pass = sum(1 for r in results if r["pass"])
    by_kind: dict[str, dict[str, int]] = {}
    for r in results:
        k = r["grade_kind"]
        d = by_kind.setdefault(k, {"pass": 0, "n": 0})
        d["n"] += 1
        d["pass"] += int(r["pass"])
    mlx_rows = [r for r in results if r["generated_tokens"] > 0]
    summary = {
        "suite": suite_name,
        "n_cases": n,
        "n_pass": n_pass,
        "pass_rate": round(n_pass / n, 4) if n else 0.0,
        "smoke": is_smoke,
        "by_kind": by_kind,
        "latency": {
            "mean_ttft_s": round(sum(r["ttft_s"] for r in mlx_rows) / len(mlx_rows), 4) if mlx_rows else 0.0,
            "mean_total_s": round(sum(r["total_s"] for r in mlx_rows) / len(mlx_rows), 4) if mlx_rows else 0.0,
            "mean_tps": round(sum(r["tps"] for r in mlx_rows) / len(mlx_rows), 2) if mlx_rows else 0.0,
            "mean_gen_tokens": round(sum(r["generated_tokens"] for r in mlx_rows) / len(mlx_rows), 1) if mlx_rows else 0.0,
            "peak_rss_mb": round(max((r["peak_rss_mb"] for r in mlx_rows), default=0.0), 1),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
