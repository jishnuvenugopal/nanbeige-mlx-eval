"""Command-line interface: ``nanbeige-mlx-eval``.

``list-suites`` / ``validate-suite`` for harness readiness, ``run`` for
execution, ``report`` / ``compare`` / ``regrade`` for idempotent
artifact-driven reporting, plus ``convert`` (passthrough to ``mlx_nanbeige``)
and ``parity`` (the MLX-vs-HF fidelity gate, Half A).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .compare import write_compare
from .report import write_report
from .runtime import MLXRuntime, MockRuntime, run_suite
from .suite import SuiteError, builtin_suites_dir, list_builtin_suites, load_suite

# `to_mlx` (mlx_nanbeige.convert) and `run_parity` (.parity) pull in mlx / mlx_lm
# at module top level. Importing them here would make every subcommand --
# including the model-free ones (list-suites, validate-suite, report, compare,
# run --runtime mock) -- require mlx, which has no linux x86_64 wheel and so
# cannot run in CI. Import them lazily inside cmd_convert / cmd_parity instead,
# matching the lazy imports already used by cmd_trace / cmd_crosscheck / cmd_bisect.


def _resolve_suite(suite: str) -> Path:
    p = Path(suite)
    if p.exists():
        return p
    for cand in list_builtin_suites():
        if cand.name == suite or cand.stem == suite:
            return cand
    cand = builtin_suites_dir() / f"{suite}.json"
    if cand.exists():
        return cand
    raise SuiteError(f"suite not found: {suite} (looked in {builtin_suites_dir()})")


def cmd_list_suites(_args):
    for p in list_builtin_suites():
        try:
            s = load_suite(p)
            print(f"{p.name}\t{s.get('language', '?')}\t{len(s['cases'])} cases\t{s.get('category', '')}")
        except SuiteError as e:
            print(f"{p.name}\tINVALID: {e}")


def cmd_validate(args):
    try:
        s = load_suite(args.path)
        print(f"OK: {s['suite']} — {len(s['cases'])} cases ({s.get('language', '?')})")
    except SuiteError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args):
    suite_path = _resolve_suite(args.suite)
    if args.runtime == "mlx":
        if not args.model:
            print("--model is required for the mlx runtime", file=sys.stderr)
            sys.exit(2)
        runtime = MLXRuntime(args.model, default_max_tokens=args.max_tokens)
    else:
        runtime = MockRuntime()
    run_dir = run_suite(
        suite_path, runtime, args.output_root,
        limit=args.limit, warmup=args.warmup, repeats=args.repeats,
    )
    print(run_dir)


def cmd_report(args):
    p = write_report(args.run_dir)
    print(p)


def cmd_compare(args):
    p = write_compare(args.run_a, args.run_b, args.output)
    print(p)


def cmd_regrade(args):
    from .regrade import regrade

    suite_path = _resolve_suite(args.suite)
    regrade(args.run_dir, suite_path, require_answer=not args.no_require_answer)
    print(f"regraded {args.run_dir}")


def cmd_trace(args):
    from .trace import run_trace

    run_trace(args.src, output=args.out, device=args.device, dtype=args.dtype)


def cmd_crosscheck(args):
    from .crosscheck import (
        render_markdown,
        render_replay_markdown,
        run_crosscheck,
        run_replay,
    )

    if args.replay:
        out = run_replay(
            args.src, prompt=args.prompt, dtype=args.dtype, output=args.out
        )
        print(render_replay_markdown(out))
        return

    out = run_crosscheck(
        args.src, prompt=args.prompt, dtype=args.dtype, output=args.out
    )
    print(render_markdown(out))
    if args.gate and "UNRESOLVED" not in out["verdict"]:
        pass  # a named side is a successful diagnosis, not a failure


def cmd_bisect(args):
    from .bisect import (
        render_markdown,
        render_sweep_markdown,
        run_bisect,
        run_scale_sweep,
    )

    if args.sweep:
        out = run_scale_sweep(
            args.src, layer_idx=args.layer, dtype=args.dtype,
            seq_len=args.seq_len, seed=args.seed, reference=args.reference,
            output=args.out,
        )
        print(render_sweep_markdown(out))
        return

    r = run_bisect(
        args.src,
        layer_idx=args.layer,
        dtype=args.dtype,
        seq_len=args.seq_len,
        seed=args.seed,
        bf16_rope=args.bf16_rope,
        input_mode=args.input,
        target_rms=args.rms,
        prompt=args.prompt,
        reference=args.reference,
        output=args.out,
    )
    print(render_markdown(r))
    if args.gate and r["first_divergent_stage"] is not None:
        print(f"FAIL: first divergent stage = {r['first_divergent_stage']}", file=sys.stderr)
        sys.exit(1)


def cmd_convert(args):
    from mlx_nanbeige.convert import to_mlx

    out = to_mlx(args.src, args.out, args.bits, args.group_size)
    print(out)


def cmd_parity(args):
    from .parity import run_parity

    r = run_parity(
        args.src, output=args.out, device=args.device, dtype=args.dtype, seed=args.seed
    )
    summary = {k: v for k, v in r.items() if k != "per_prompt"}
    print(json.dumps(summary, indent=2))
    # Fidelity gate (P2.4): the docstring says the bar is "cosine ~= 1"; enforce
    # it on cpu where bf16 isn't the limiting factor. Exit non-zero if it fails.
    if args.gate is not None:
        mean_cos = r.get("mean_cosine", 0.0)
        if mean_cos < args.gate:
            print(
                f"FIDELITY GATE FAILED: mean_cosine={mean_cos:.4f} < {args.gate}",
                file=sys.stderr,
            )
            sys.exit(1)


def cmd_smoke(args):
    """A quick, gated real-model smoke: one short generation (not a benchmark)."""
    from mlx_lm import generate, load

    model, tokenizer = load(args.model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
    )
    out = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
    print(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="nanbeige-mlx-eval",
        description="Bilingual agentic-readiness eval for the Nanbeige4.2-3B MLX port.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-suites", help="list packaged suites").set_defaults(func=cmd_list_suites)

    v = sub.add_parser("validate-suite", help="schema-validate a suite JSON")
    v.add_argument("path")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="run a suite over a runtime")
    r.add_argument("--suite", required=True)
    r.add_argument("--runtime", choices=["mock", "mlx"], default="mock")
    r.add_argument("--model", help="MLX model repo (required for --runtime mlx)")
    r.add_argument("--output-root", default="benchmark_results")
    r.add_argument("--limit", type=int, default=None, help="cap case count (marks run smoke-only)")
    r.add_argument("--max-tokens", type=int, default=768)
    r.add_argument("--warmup", type=int, default=0,
                   help="run & discard case 0 N times first (absorbs Metal/compile cost)")
    r.add_argument("--repeats", type=int, default=1,
                   help="repeat each case N times and take the median of timing fields")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="regenerate report.md from a run dir")
    rep.add_argument("run_dir")
    rep.set_defaults(func=cmd_report)

    cmp = sub.add_parser("compare", help="diff two run dirs")
    cmp.add_argument("run_a")
    cmp.add_argument("run_b")
    cmp.add_argument("--output", default=None)
    cmp.set_defaults(func=cmd_compare)

    rg = sub.add_parser("regrade", help="re-grade a persisted run against its suite")
    rg.add_argument("run_dir")
    rg.add_argument("--suite", required=True, help="suite name or path the run used")
    rg.add_argument("--no-require-answer", action="store_true",
                    help="grade the whole stream (opt out of reasoning isolation)")
    rg.set_defaults(func=cmd_regrade)

    cv = sub.add_parser("convert", help="convert Nanbeige HF -> MLX quant")
    cv.add_argument("--src", required=True)
    cv.add_argument("--out", required=True)
    cv.add_argument("--bits", type=int, required=True)
    cv.add_argument("--group-size", type=int, default=64)
    cv.set_defaults(func=cmd_convert)

    par = sub.add_parser("parity", help="Half A: MLX port vs HF reference logits")
    par.add_argument("--src", required=True, help="local Nanbeige HF repo")
    par.add_argument("--out", default=None)
    par.add_argument("--device", choices=["cpu", "mps"], default="cpu",
                     help="device for the HF reference (default cpu; bf16/CPU avoids MPS noise)")
    par.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                     help="dtype for both sides (bf16/CPU ~= 8.3 GB; fits 16 GB)")
    par.add_argument("--seed", type=int, default=0,
                     help="seed for any sampling in the parity run (rope_precision "
                          "uses real q and is reproducible regardless)")
    par.add_argument("--gate", type=float, default=None,
                     help="exit non-zero if mean_cosine falls below this (e.g. 0.99)")
    par.set_defaults(func=cmd_parity)

    tr = sub.add_parser("trace", help="per-effective-layer divergence trace (44 layers)")
    tr.add_argument("--src", required=True, help="local Nanbeige HF repo")
    tr.add_argument("--out", default="trace.json")
    tr.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    tr.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    tr.set_defaults(func=cmd_trace)

    bi = sub.add_parser(
        "bisect",
        help="single-layer, stage-by-stage divergence bisect (fp32-capable, ~1.2 GB)",
    )
    bi.add_argument("--src", required=True, help="local Nanbeige HF repo")
    bi.add_argument("--layer", type=int, default=0, help="physical layer index to bisect")
    bi.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="fp32 is the decisive run: a gap there is logic, not numerics")
    bi.add_argument("--seq-len", type=int, default=8)
    bi.add_argument("--seed", type=int, default=0)
    bi.add_argument("--bf16-rope", action="store_true",
                    help="apply the reference's bf16 cos/sin downcast on BOTH sides, "
                         "to isolate that effect")
    bi.add_argument("--input", choices=["real", "scaled", "random"], default="real",
                    help="probe input. 'real' = actual embed_tokens output (rms~0.024, "
                         "the only regime valid for bf16 precision claims). "
                         "'random' is unit-variance and ~42x too large.")
    bi.add_argument("--rms", type=float, default=None,
                    help="target input RMS for --input scaled")
    bi.add_argument("--prompt", default=None, help="prompt for --input real")
    bi.add_argument("--reference", choices=["real", "mirror"], default="real",
                    help="what to compare the port against. 'real' instantiates the "
                         "checkpoint's own NanbeigeDecoderLayer (the arbiter). "
                         "'mirror' uses _torch_stages, a reimplementation — "
                         "agreement there does NOT prove agreement with Nanbeige's code.")
    bi.add_argument("--sweep", action="store_true",
                    help="sweep input RMS from 1.0 down to the real embedding scale "
                         "and report block cosine at each -- reconciles bisect vs trace")
    bi.add_argument("--out", default=None)
    bi.add_argument("--gate", action="store_true",
                    help="exit non-zero if any stage diverges")
    bi.set_defaults(func=cmd_bisect)

    xc = sub.add_parser(
        "crosscheck",
        help="reconcile bisect vs trace: 4x4 layer-0 matrix + embedding check",
    )
    xc.add_argument("--src", required=True, help="local Nanbeige HF repo")
    xc.add_argument("--prompt", default="The capital of France is")
    xc.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    xc.add_argument("--out", default=None)
    xc.add_argument("--gate", action="store_true")
    xc.add_argument("--replay", action="store_true",
                    help="capture the model's real layer-0 call arguments and replay "
                         "them standalone — decides whether B or D is the layer's true "
                         "behaviour, and therefore whether the port is actually cleared")
    xc.set_defaults(func=cmd_crosscheck)

    sm = sub.add_parser("smoke", help="gated one-shot real-model generation")
    sm.add_argument("--model", required=True)
    sm.add_argument("--prompt", default="Say hello.")
    sm.add_argument("--max-tokens", type=int, default=64)
    sm.set_defaults(func=cmd_smoke)

    return ap


def main(argv: Optional[list[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
