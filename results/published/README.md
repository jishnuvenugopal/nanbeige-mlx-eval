# Published results — provenance

This directory holds the committed artifacts the README and model cards cite, so
every number is checkable from the repo without re-running the model:

- `parity_cpu_bf16.json`, `trace_cpu_bf16.json` — Half A (fidelity) measurements.
- `bisect_*.json`, `crosscheck_layer0.json`, `replay_layer0.json` — the
  differential-testing tooling that produced the ruled-out-cause table.
- `agentic_{en,zh}-mlx-*` — the six ladder runs (4 / 6 / 8-bit × EN / ZH) for
  Half B, each with `manifest.json`, `results.jsonl`, `summary.json`, `report.md`.

## On the `git_commit` field

Every ladder manifest stamps `git_commit: c800313` ("Settle parity: fp32 bisect
verifies arithmetic; widen suites to 30"). That is the commit the runs were
executed at — **not** the `v0.2.0` release commit (the release commit came later,
on the `fix/parity-bisect-and-suite-widen` branch).

This is intentional and the numbers are still valid: between `c800313` and the
release commit, the only change to code that affects inference is a **comment**
in `nanbeige_mlx/nanbeige_mlx/model.py` documenting a measured no-op (the RoPE
fp32 upcast experiment — see Addendum 6 of
[`docs/investigation-log.md`](../../docs/investigation-log.md)). No executable
line that the model runs changed, so the Half B pass rates, latencies, and
memory numbers are reproducible from either commit. Re-running on the tag would
produce identical results and was not done to avoid rewriting ~8 GB of identical
artifacts.

## `--repeats`

EN@4-bit (`agentic_en-mlx-20260726T071154Z`) was run at `--repeats 3`; the other
five ladder runs at `--repeats 1`. Pass rate is repeat-invariant — `--repeats`
only stabilizes the timing fields. The README states this per-config rather than
assuming a uniform setting.

## Regenerating

All tools here read **only** persisted files and never re-run the model:
`nanbeige-mlx-eval report <run_dir>` regenerates `report.md` idempotently,
`regrade <run_dir> --suite <name>` retroactively applies the current graders.
To reproduce a run from scratch, see the CLI workflow in the root
[`README.md`](../../README.md).
