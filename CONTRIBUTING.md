# Contributing

Contributions welcome — bug reports, suite cases, grader improvements, port
fixes, and docs all help. This is an independent project; it is not affiliated
with or endorsed by the Nanbeige team.

## The non-obvious thing: you do not need the model

The full Nanbeige4.2-3B checkpoint is ~8 GB and the converted quants another
~10 GB across 4/6/8-bit. **Most contributions need none of it.** The harness
ships a **mock runtime** that exercises the entire suite → grader → report →
manifest pipeline without a model download:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q                                   # 30 tests, no model, no Metal
.venv/bin/nanbeige-mlx-eval run --suite smoke --runtime mock --output-root /tmp/out
```

The mock runtime is in `nanbeige_mlx_eval/runtime.py`; the model-free tests are
under `tests/` (`test_grading`, `test_manifest`, `test_mock_runtime`,
`test_regrade`, `test_suite_schema`). Start here for harness work.

## What lives where

| area | path | needs model? |
|---|---|---|
| eval harness | `nanbeige_mlx_eval/` | no (mock) / yes (real run) |
| the MLX port | `mlx_nanbeige/mlx_nanbeige/` | yes (to exercise) |
| suites | `suites/` (`agentic_en`, `agentic_zh`, `smoke`) | no (schema) |
| graders | `nanbeige_mlx_eval/grading.py` | no |
| fidelity tooling | `nanbeige_mlx_eval/{parity,trace,bisect,crosscheck}.py` | yes (HF reference) |
| published results | `results/published/` | no (read-only) |

## Before you open a PR

1. **Tests pass.** `.venv/bin/pytest -q` must be green. If you touch the port,
   also run `mlx_nanbeige/tests/test_cache_consistency.py` (needs MLX).
2. **Suites validate.** `nanbeige-mlx-eval validate-suite suites/<your>.json`
   against the schema in `nanbeige_mlx_eval/suite.py`.
3. **Numbers stay checkable.** If you cite a measurement in a README or model
   card, the artifact that produced it lives under `results/published/` (or is
   reproducible by a documented command). Don't state a number with no trail.
4. **Mocks first.** If you add a harness feature, add a mock-runtime test so CI
   can exercise it without a 16 GB machine.

## Adding a suite case

Suite cases live in `suites/agentic_{en,zh}.json` (30 each) and follow the
schema enforced by `validate-suite`. Each case offers one or more tools in the
Hermes/ChatML `<tool_call>` format and a natural-language request; the grader
grades **only the answer tail** (text after the first `</think>`), so arguments
should be deterministic (city names, emails, ISO dates, IANA timezones). Run
`validate-suite` after editing.

## Reporting a bug

The most useful bug reports include:

- `nanbeige-mlx-eval` version and `mlx.core.__version__`
- the exact command and its output
- for conversion failures, the full `mlx-nanbeige-convert` log

## Honesty about results

This project documents a known, open end-to-end logit gap against the HF
reference (mean cosine 0.847) alongside what *is* verified — see
[`docs/investigation-log.md`](docs/investigation-log.md). Contributions that
narrow or close that gap are especially welcome; contributions that paper over
it are not. If your change moves a fidelity number, record the before/after and
which commit produced each.
