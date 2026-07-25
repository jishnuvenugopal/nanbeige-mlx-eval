"""Command-line interface: ``nanbeige-mlx-eval``.

Subcommands mirror the discipline of the sibling ``ornith-mlx-eval`` project:
``list-suites`` / ``validate-suite`` for harness readiness, ``run`` for
execution, ``report`` / ``compare`` for idempotent artifact-driven reporting,
plus the project-specific ``convert`` (HF -> MLX quant) and ``parity``
(the MLX-vs-HF fidelity gate, Half A).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .compare import write_compare
from .convert import to_mlx
from .parity import run_parity
from .report import write_report
from .runtime import MLXRuntime, MockRuntime, run_suite
from .suite import SuiteError, builtin_suites_dir, list_builtin_suites, load_suite


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
        suite_path, runtime, args.output_root, limit=args.limit
    )
    print(run_dir)


def cmd_report(args):
    p = write_report(args.run_dir)
    print(p)


def cmd_compare(args):
    p = write_compare(args.run_a, args.run_b, args.output)
    print(p)


def cmd_convert(args):
    out = to_mlx(args.src, args.out, args.bits, args.group_size)
    print(out)


def cmd_parity(args):
    r = run_parity(args.src, output=args.out)
    summary = {k: v for k, v in r.items() if k != "per_prompt"}
    print(json.dumps(summary, indent=2))


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
    ap = argparse.ArgumentParser(prog="nanbeige-mlx-eval", description=__doc__)
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
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="regenerate report.md from a run dir")
    rep.add_argument("run_dir")
    rep.set_defaults(func=cmd_report)

    cmp = sub.add_parser("compare", help="diff two run dirs")
    cmp.add_argument("run_a")
    cmp.add_argument("run_b")
    cmp.add_argument("--output", default=None)
    cmp.set_defaults(func=cmd_compare)

    cv = sub.add_parser("convert", help="convert Nanbeige HF -> MLX quant")
    cv.add_argument("--src", required=True)
    cv.add_argument("--out", required=True)
    cv.add_argument("--bits", type=int, required=True)
    cv.add_argument("--group-size", type=int, default=64)
    cv.set_defaults(func=cmd_convert)

    par = sub.add_parser("parity", help="Half A: MLX port vs HF reference logits")
    par.add_argument("--src", required=True, help="local Nanbeige HF repo")
    par.add_argument("--out", default=None)
    par.set_defaults(func=cmd_parity)

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
