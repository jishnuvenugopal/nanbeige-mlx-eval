"""Run-times and the run orchestrator.

Two run-times share a common ``run_case`` interface:

* :class:`MockRuntime` – the default. Deterministic, requires no model. It
  returns a canned, expectation-shaped response so the full pipeline (grading,
  artifact writing, reporting) can be exercised in CI or triage without a
  download. It never makes a capability *claim* about the real model.

* :class:`MLXRuntime` – the real thing. Loads a converted MLX repo and generates
  with an explicit greedy sampler for reproducibility. Gated behind an explicit
  model path (there is no ambient "download on demand": you must point at a repo
  you converted).

The orchestrator :func:`run_suite` executes a suite over a run-time and writes
the standard four artifacts (manifest / results / summary / report). Runs are
**smoke-flagged** when ``limit`` caps the case count, which disables any
benchmark-quality framing in downstream reports.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional

from . import profile
from .grading import grade
from .suite import SuiteError, load_suite, validate_suite


@dataclass
class CaseResult:
    output: str
    profile: Optional[profile.GenProfile] = None
    meta: dict[str, Any] = field(default_factory=dict)


class MockRuntime:
    """Deterministic, model-free run time for harness plumbing."""

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
            # json_schema grader is exercised on the mock path. The mock emits
            # it as a direct answer (no <think> block), matching a non-reasoning
            # model — grading treats the whole stream as the answer.
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
        return CaseResult(
            output=out,
            meta={"runtime": "mock", "stop_reason": "stop"},
        )


class MLXRuntime:
    """Real MLX run time. Explicit greedy sampler; thinking is controllable."""

    name = "mlx"

    def __init__(self, model_path: str, default_max_tokens: int = 768):
        # Imported lazily so the harness imports without mlx_lm at collection time.
        from mlx_lm import load, stream_generate  # noqa: F401
        from mlx_lm.sample_utils import make_sampler

        self._stream_generate = stream_generate
        # Greedy is *asserted*, not assumed: the manifest records "greedy", so
        # the sampler must match. mlx-lm's default isn't guaranteed to be greedy.
        self._sampler = make_sampler(temp=0.0)
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
        # Thinking control (B9): per-case override of the reasoning block. When
        # explicitly disabled, the template omits the leading <think>.
        if "enable_thinking" in case:
            kwargs["enable_thinking"] = bool(case["enable_thinking"])
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def run_case(self, case: dict[str, Any]) -> CaseResult:
        prompt = self._build_prompt(case)
        max_tokens = case.get("max_tokens", self.default_max_tokens)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        text_parts: list[str] = []
        n = 0
        finish_reason: Optional[str] = None
        ttft: Optional[float] = None
        # Per-case allocator peak (B4): reset before the measured region.
        profile.reset_peak_memory()
        t0 = profile.now()
        for resp in self._stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=self._sampler,
        ):
            if n == 0:
                ttft = profile.now() - t0
            text_parts.append(resp.text)
            n += 1
            if getattr(resp, "finish_reason", None):
                finish_reason = resp.finish_reason
        total = profile.now() - t0

        gen = profile.GenProfile(
            prompt_tokens=len(prompt_ids),
            generated_tokens=n,
            total_seconds=total,
            ttft_seconds=ttft or 0.0,
            peak_rss_mb=profile.mlx_peak_mb(),
        )
        return CaseResult(
            output="".join(text_parts),
            profile=gen,
            meta={
                "runtime": "mlx",
                "model": str(self.model_path),
                "sampler": "greedy(temp=0.0)",
                "enable_thinking": case.get("enable_thinking"),
                "stop_reason": finish_reason or "length",
            },
        )


def _pkg_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "absent"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _env_info() -> dict[str, Any]:
    # mlx.core.__version__ (not mlx.__version__) when mlx is importable. The
    # import is best-effort: this function runs on the mock-runtime path (which
    # the model-free tests exercise) and on CI runners where mlx may not be
    # installable (no linux x86_64 wheel). Stamp "unavailable" rather than
    # hard-failing so a missing optional dep can't break manifest writing.
    try:
        import mlx.core as mx
        mlx_version: Any = mx.__version__
    except Exception:
        mlx_version = "unavailable"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": mlx_version,
        "mlx_lm": _pkg_version("mlx-lm"),
        "nanbeige_mlx": _pkg_version("nanbeige-mlx"),
        "transformers": _pkg_version("transformers"),
        "git_commit": _git_sha(),
    }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _quant_info(model_path: str | None) -> Optional[dict[str, Any]]:
    if not model_path:
        return None
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    return {"bits": q.get("bits"), "group_size": q.get("group_size")} or None


def run_suite(
    suite_path: str | Path,
    runtime,
    output_root: str | Path,
    *,
    limit: Optional[int] = None,
    run_id: Optional[str] = None,
    warmup: int = 0,
    repeats: int = 1,
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

    # Warmup (B6): run and discard case 0 so first-call Metal kernel + mx.compile
    # cost doesn't land in the published numbers.
    for _ in range(warmup):
        runtime.run_case(cases[0])

    suite_default_require_answer = suite.get("require_answer")
    suite_require_answer: Optional[bool] = suite_default_require_answer

    results: list[dict[str, Any]] = []
    for case in cases:
        # Per-case repeats -> median of timing fields (B6); pass/detail come from
        # the final repeat's output (greedy, so outputs are identical anyway).
        repeat_profiles: list[profile.GenProfile] = []
        cr: CaseResult = CaseResult(output="", meta={})
        for _rep in range(max(1, repeats)):
            cr = runtime.run_case(case)
            if cr.profile:
                repeat_profiles.append(cr.profile)

        require_answer = case.get("require_answer", suite_require_answer)
        verdict = grade(case["expect"], cr.output, require_answer=bool(require_answer))

        if repeat_profiles:
            tps_vals = [p.tokens_per_second for p in repeat_profiles]
            ttft_vals = [p.ttft_seconds for p in repeat_profiles]
            total_vals = [p.total_seconds for p in repeat_profiles]
            p = repeat_profiles[-1]
            row_tps = round(statistics.median(tps_vals), 2)
            row_ttft = round(statistics.median(ttft_vals), 4)
            row_total = round(statistics.median(total_vals), 4)
            prompt_tokens = p.prompt_tokens
            gen_tokens = p.generated_tokens
            peak = p.peak_rss_mb
        else:
            row_tps = row_ttft = row_total = 0.0
            prompt_tokens = gen_tokens = 0
            peak = 0.0

        results.append(
            {
                "id": case["id"],
                "pass": verdict["pass"],
                "detail": verdict["detail"],
                "grade_kind": case["expect"]["type"],
                "stop_reason": cr.meta.get("stop_reason"),
                "output_hash": _hash_text(cr.output),
                "output": cr.output,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": gen_tokens,
                "ttft_s": row_ttft,
                "total_s": row_total,
                "tps": row_tps,
                "peak_rss_mb": round(peak, 1),
            }
        )

    _write_manifest(
        run_dir, suite, runtime, run_id, len(cases), is_smoke, limit,
        warmup=warmup, repeats=repeats,
    )
    _write_results(run_dir, results)
    _write_summary(run_dir, results, suite_name, is_smoke)

    from .report import write_report

    write_report(run_dir)
    return run_dir


def _write_manifest(run_dir, suite, runtime, run_id, n_cases, is_smoke, limit, *, warmup, repeats):
    model_path = getattr(runtime, "model_path", None)
    # Suite content hash so a run is traceable to the exact suite bytes.
    suite_sha = None
    try:
        suite_src = Path(suite.get("_source_path", "")) if isinstance(suite, dict) else None
    except Exception:
        suite_src = None
    manifest = {
        "run_id": run_id,
        "suite": suite["suite"],
        "suite_language": suite.get("language"),
        "suite_category": suite.get("category"),
        "runtime": runtime.name,
        "model": model_path,
        "quantization": _quant_info(model_path),
        "env": _env_info(),
        "settings": {
            "decoding": "greedy(temp=0.0)" if runtime.name == "mlx" else "n/a",
            "default_max_tokens": getattr(runtime, "default_max_tokens", None),
            "limit": limit,
            "smoke": is_smoke,
            "warmup": warmup,
            "repeats": repeats,
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
    ci_lo, ci_hi = profile.wilson(n_pass, n)

    # Throughput as a ratio of totals (B3), not a mean of per-case ratios.
    if mlx_rows:
        gen_tokens = sum(r["generated_tokens"] for r in mlx_rows)
        decode_time = sum(max(r["total_s"] - r["ttft_s"], 0.0) for r in mlx_rows)
        decode_tps_aggregate = round(gen_tokens / decode_time, 2) if decode_time else 0.0
        tps_list = [r["tps"] for r in mlx_rows]
        ttft_list = [r["ttft_s"] for r in mlx_rows]
        # TTFT split by regime (B3): tool-use prompts vs bare prompts.
        with_tools = [r["ttft_s"] for r in mlx_rows if r.get("prompt_tokens", 0) > 64]
        bare = [r["ttft_s"] for r in mlx_rows if r.get("prompt_tokens", 0) <= 64]
        latency = {
            "decode_tps_aggregate": decode_tps_aggregate,
            "decode_tps_median": round(statistics.median(tps_list), 2),
            "ttft_s_median": round(statistics.median(ttft_list), 3),
            "ttft_s_by_regime": {
                "with_tools": round(statistics.mean(with_tools), 3) if with_tools else None,
                "bare_prompt": round(statistics.mean(bare), 3) if bare else None,
            },
            "mean_gen_tokens": round(statistics.mean([r["generated_tokens"] for r in mlx_rows]), 1),
            "peak_rss_mb": round(max((r["peak_rss_mb"] for r in mlx_rows), default=0.0), 1),
        }
    else:
        latency = {}

    # Truncation visibility (P1.2): which cases stopped on length, not eos.
    truncated_ids = [r["id"] for r in results if r.get("stop_reason") == "length"]

    summary = {
        "suite": suite_name,
        "n_cases": n,
        "n_pass": n_pass,
        "pass_rate": round(n_pass / n, 4) if n else 0.0,
        "pass_rate_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "smoke": is_smoke,
        "by_kind": by_kind,
        "truncated": {"n": len(truncated_ids), "ids": truncated_ids},
        "latency": latency,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
