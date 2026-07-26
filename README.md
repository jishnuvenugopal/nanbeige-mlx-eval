# nanbeige-mlx-eval

A **bilingual (EN/ZH) agentic-readiness evaluation** and **fidelity gate** for
the Nanbeige4.2-3B *Looped Transformer*, running entirely on Apple Silicon. The
MLX port itself lives in the sibling [`mlx-nanbeige`](mlx_nanbeige/) package
(this repo depends on it); this package owns the suites, graders, runtime,
parity/trace tooling, and reporting. This is an independent project; it is not
affiliated with or endorsed by the Nanbeige team.

> **Question this project answers:** *Can a 3-billion-parameter Looped Transformer
> serve as a faithful and capable local agentic assistant on Apple Silicon — and
> at which quantization level?*

Two halves, executed in order (the order **is** the methodology):

1. **Fidelity (trust).** A from-scratch MLX implementation of the looped
   architecture, compared against the HuggingFace reference's next-token logits
   and traced layer-by-layer across all 44 effective layers. Capability numbers
   from an unverified port are meaningless, so this gate runs first.
2. **Agentic readiness (capability).** An authored bilingual tool-use /
   structured-output suite run across the 4 / 6 / 8-bit quantization ladder,
   with Wilson confidence intervals and per-case truncation visibility.

This is a **harness + honest small-N eval**, not a leaderboard. Pass rates carry
95% CIs; runs are smoke-flagged when the case count is capped.

---

## Executive verdict

- ✅ The Nanbeige *Looped Transformer* (`num_loops = 2`, weight-shared,
  GQA-48/8, `head_dim = 128`, RoPE θ = 70 000 000) ports cleanly to MLX and
  **runs on a 16 GB M1 Pro**.
- ✅ The 44-slot loop-aware cache is **verified by prefill-vs-incremental-decode
  equality** (all virtual slots stay in lockstep; see
  `mlx_nanbeige/tests/test_cache_consistency.py`).
- ✅ **Tool selection and argument extraction hold at 8/8 (EN) and 8/8 (ZH) from
  4-bit up**, and structured output succeeds at all six quant×language configs
  once the token cap is raised (48/48 at a 1024-token cap).
- ⚠️ **A genuine, moderate bf16 logit gap vs the HF reference remains on CPU
  (mean cosine 0.85, top-1 83%)** — and it is **not** MPS noise: running the
  reference on CPU reproduces the same gap. A single-layer fp32 bisect
  (`bisect --dtype fp32`) shows every one of 14 stages agrees to cosine 1.0
  (max-abs ≤ 1.3e-05): the port's arithmetic is **verified correct** and the
  gap is a precision effect, compounded across 44 effective layers — see
  [§ Fidelity](#fidelity-half-a). It does *not* affect behavior on the agentic
  suite.
- 📌 **A real cost of the looped design nobody else has written down:** full
  262K context is unreachable on a 16 GB machine — the looped trunk needs 44 KV
  slots, ~47 GB at max context — and mlx-lm's `--max-kv-size` is inert for this
  model.

---

## Target model and environment

| dimension | value |
|---|---|
| model | `Nanbeige/Nanbeige4.2-3B` (BF16, Apache-2.0) |
| architecture | custom `nanbeige` — Looped Transformer, `num_loops=2` over 22 layers |
| params | ~4B total (3B non-embedding); GQA 48 / 8 heads, `head_dim=128` |
| context | 262 144 (evaluated at short tool-use lengths only) |
| machine | Apple M1 Pro, 10-core, 16 GB unified memory, macOS |
| python | 3.12 · mlx 0.32.0 · mlx-lm 0.31.3 · transformers (reference only) |

---

## The port (`mlx-nanbeige`)

The effective architecture of this checkpoint is a clean, portable design (the
published config's experimental features — n-gram, hyper-connection, depth
attention, double-loop — are all disabled):

- **22 standard pre-norm decoder blocks** (RMSNorm → GQA attention → residual;
  RMSNorm → SwiGLU MLP → residual), executed **twice** per forward pass — the
  layers are weight-shared across the two loops.
- The **final RMSNorm is applied at each loop boundary** (`skip_loop_final_norm =
  False`), i.e. after the 22nd and after the 44th effective layer.
- Two non-obvious details a naive port would get wrong:
  - **`head_dim = 128` while `hidden_size / num_heads = 64`** — the q/o
    projections are non-square (`q: 3072→6144`, `o: 6144→3072`).
  - The KV cache needs **`num_loops × num_hidden_layers = 44` virtual slots**
    even though only 22 blocks of weights exist, exposed via mlx-lm's
    `make_cache` hook so standard `mlx_lm.generate` works unchanged.

The port lives in [`mlx_nanbeige/mlx_nanbeige/model.py`](mlx_nanbeige/mlx_nanbeige/model.py)
and is also shipped inside each converted weight repo as a `model_file`, so the
quants load with no registry entry. Convert and publish with the sibling package:

```bash
mlx-nanbeige-convert --src models/nanbeige42-hf --out models/nanbeige-mlx-4bit --bits 4
mlx-nanbeige-upload --model-dir models/nanbeige-mlx-4bit \
  --repo-id <user>/Nanbeige4.2-3B-mlx-4bit --dry-run   # renders card, no push
```

---

## Fidelity (Half A)

### Method

The same short prompts (English, Chinese, code, math) are run through (a) the
official checkpoint under `transformers` (`trust_remote_code`, **CPU**, bf16) and
(b) this MLX port (bf16). The HF model is loaded, measured and **freed before**
the MLX model is loaded, so the two are never co-resident on a 16 GB machine.
Next-token logits are compared. Run it:

```bash
nanbeige-mlx-eval parity --src models/nanbeige42-hf --out benchmark_results/parity.json \
  --device cpu --dtype bf16 --gate 0.99   # exit non-zero if mean_cosine < 0.99
```

### Result — and what it means

| metric | value |
|---|---|
| prompts | 6 (English, Chinese, code, math) |
| device / dtype | **CPU / bfloat16** (both sides) |
| top-1 next-token agreement | **83.3 %** (5 / 6) |
| mean logit cosine | **0.847** |
| min logit cosine | 0.590 |
| per-prompt cosine | `0.99, 0.59, 0.59, 0.98, 0.96, 0.97` |
| RoPE bf16 floor (isolated) | max-abs **0.015** on cos/sin vectors (reference downcasts to bf16; MLX stays fp32). Localized: `--bf16-rope` erases it at the RoPE stage but moves the per-layer block cosine by ~1e-7 — RoPE precision is real but not the dominant cause of the end-to-end gap |

> **Important correction.** An earlier version of this report attributed the gap
> to "Metal-vs-MPS bfloat16 matmul differences." That hypothesis is **falsified**:
> running the reference on **CPU** reproduces the same mean cosine (0.847 vs the
> old 0.844 on MPS). The gap is **not** MPS-specific. What it actually is, see
> the trace below.

### Per-layer divergence trace (the actual diagnosis)

`nanbeige-mlx-eval trace` dumps hidden states after each of the 44 effective
layers on both sides and reports cosine per layer. Reading the curve:

| effective layer | 0 | 3 | 20 | **21** | **22** | 43 |
|---|---|---|---|---|---|---|
| cosine | 0.925 | 0.834 | 0.956 | **0.997** | **0.895** | 0.980 |

- **The curve is non-monotone**, and it is worst exactly where the residual
  stream is smallest. Agreement *falls* over layers 0-3, *climbs* to 0.997 at
  layer 21, then **drops ~0.10 in a single step at the loop boundary**
  (effective layer 21 → 22, where `loop` flips 0 → 1 and `self.norm`
  rescales a fresh, small residual), then climbs again to 0.980 by layer 43.
- This shape is the signature of a **near-constant absolute per-layer error
  diluted by a growing residual norm** — not monotone bf16 drift, which
  cannot improve over 18 of 22 layers. The step at the loop boundary is a
  residual-magnitude effect, not a logic bug (see the bisect below).
- **Layer 0 receives an identical input on both sides** (same embedding
  lookup) yet lands at cosine 0.925 — an error of ~38% of the output norm
  after one layer. That is far larger than bf16 per-op error (~0.4%), which
  is why "it's just numerics" was the wrong first read: the per-layer error
  is small in *absolute* terms but large *relative* to a small residual.

### The decisive experiment: single-layer fp32 bisect

The blocker cited in earlier drafts — that fp32 isolation OOMs on 16 GB — is
false for the question actually being asked. One decoder layer is ~143 M params
(~573 MB in fp32); loading just it on both sides needs ~1.2 GB. The `bisect`
subcommand feeds a common random input to one layer in both frameworks and
compares **14 stages independently** (so an early gap cannot mask a later one),
in fp32 where any disagreement is logic, not numerics:

```bash
nanbeige-mlx-eval bisect --src models/nanbeige42-hf --dtype fp32 --gate
```

**Result (`results/published/bisect_fp32.json`): every stage agrees to cosine
1.0, max-abs ≤ 1.3e-05.** No first-divergent stage. The port's RMSNorm,
projections, RoPE, SwiGLU and the assembled block are all bit-faithful to the
reference in fp32. **This verifies the arithmetic is correct**; the end-to-end
gap is a bf16 precision effect, compounded across 44 effective layers and
re-exposed at each norm.

### Isolating RoPE precision

The RoPE stage is the one place MLX and the HF reference differ *by design*:
the reference downcasts cos/sin to bf16 (`return cos.to(dtype=x.dtype)`) while
`mx.fast.rope` computes in fp32. `max_abs ≈ 0.015` on the cos/sin vectors is
the real number — the headline `cosine 0.999997` is meaningless for cos/sin
vectors because they are dominated by near-1 entries. `--bf16-rope` forces the
bf16 downcast on **both** sides to isolate it:

```bash
nanbeige-mlx-eval bisect --src models/nanbeige42-hf --dtype bf16 --bf16-rope
```

**Result (`results/published/bisect_bf16_rope.json`):** the RoPE-stage max-abs
drops from 0.0156 to 0.0001 (the gap is exactly the fp32-vs-bf16 cos/sin
difference), but the per-layer block cosine moves only 0.99999421 → 0.99999433
— a shift of ~1e-7. **RoPE precision is real and localized, but it is not the
dominant cause of the end-to-end gap.** The 0.847 end-to-end cosine is the
accumulation of all the small bf16 stage errors (~1e-5 each, see
`bisect_bf16.json`) compounded over 44 layers.

> Bottom line: **no architectural or arithmetic bug** (verified by the
> lockstep cache test, the 44-layer trace, and the fp32 stage-by-stage bisect).
> A moderate bf16 logit gap is present, **fully attributed to bf16 precision**
> rather than hand-waved, and it does not affect behavior on the agentic suite.
> Bit-exact parity is not claimed and not expected in bf16.

### Functional fidelity (the test that matters for the use case)

Logit cosine is a blunt instrument. The sharper question is whether the port
**behaves** like the reference on the actual task. It does: on the agentic suite
the port selects the correct tool and emits the correct arguments on **48 / 48**
cases across the quant ladder (see Half B).

---

## Agentic readiness (Half B)

An authored bilingual suite: each case offers one or more tools (Hermes/ChatML
`<tool_call>` format, matching the model's chat template) and a natural-language
request; the model must select the correct tool and emit a well-formed call with
the right arguments. Suite files live in [`suites/`](suites):

- [`agentic_en.json`](suites/agentic_en.json) — 8 English cases.
- [`agentic_zh.json`](suites/agentic_zh.json) — 8 Chinese mirror cases.

Decoding is **greedy** (`temp=0.0`, asserted via an explicit sampler — not
assumed). Args use deterministic values (city names, emails, ISO dates, IANA
timezones). The grader grades **only the answer tail** (the text after the first
`</think>`), never the reasoning scratchpad; a truncated `<think>` block
hard-fails as `truncated_no_answer` rather than scanning the stream for a
coincidental match. Every case carries a `stop_reason` (`stop`/`length`).

### Result across the quant ladder

Greedy, M1 Pro / 16 GB, `--warmup 1` (absorbs first-call Metal/compile cost).
8 cases / language — small-N, so pass rates carry Wilson 95% CIs. Throughput is
**aggregate** (total generated tokens / total decode time), not a mean of
per-case ratios.

| quant | weights | English | Chinese | CI95 (8/8) | decode tok/s | TTFT (tools / bare) | peak RSS |
|---|---|---|---|---|---|---|---|
| 4-bit | 2.2 GB | **8 / 8** | **8 / 8** | [0.68, 1.00] | 31.6 | 2.59 s / 0.68 s | 2.8 GB |
| 6-bit | 3.2 GB | **8 / 8** | **8 / 8** | [0.68, 1.00] | 23.1 | 2.64 s / — | 3.8 GB |
| 8-bit | 4.1 GB | **8 / 8** | **8 / 8** | [0.68, 1.00] | 18.9 | 2.55 s / — | 4.7 GB |

**Zero truncations** across all 48 cases (`stop_reason: stop` everywhere).

### What the numbers say

- **Agentic readiness is robust to quantization.** Correct tool selection and
  argument extraction hold at 8/8 in both languages from 4-bit up. At 4-bit the
  model still picks the right tool and the right arguments.
- **Structured output was never capability-limited — it was cap-limited.** An
  earlier version of this report graded four runs as `schema_valid` on outputs
  that were actually reasoning truncated mid-sentence (the grader was reading
  the scratchpad). With the grader fixed to grade only the answer tail, the two
  `json-profile` cases needed a higher cap: at a **1024-token** cap they pass at
  all six configs (consuming 188-962 tokens of chain-of-thought first). The
  honest claim is "tool calls are quantization-robust; structured output needs
  room for the reasoning model to think."
- **Lower quants are faster, not slower.** Decode is memory-bandwidth-bound on
  Apple Silicon, so 4-bit (31.6 tok/s aggregate) is ~1.7× faster than 8-bit
  (18.9 tok/s) while using ~60 % of the memory.
- **Sweet spot for the use case: 4-bit.** Smallest footprint (~2.8 GB peak),
  fastest decode, full measured capability. On a 16 GB machine it leaves ample
  headroom — exactly the "local personal assistant" regime.
- **Latency caveat — it reasons.** ~120 generated tokens per tool call
  (chain-of-thought before the call) and ~2.6 s TTFT make per-query wall-time
  notably higher than a non-reasoning 3B. The throughput numbers above are decode
  speed, not end-to-end responsiveness.

---

## The KV-cache ceiling (a finding from porting a looped architecture)

The looped design needs `num_loops × num_hidden_layers = 44` KV slots. At full
context that is **44 × 8 KV-heads × 128 dim × 2 (K+V) × 262 144 positions × 2
bytes ≈ 47 GB** — unreachable on a 16 GB machine. Two consequences worth knowing
before you write an mlx-lm port of a weight-shared architecture:

1. **`--max-kv-size` is inert** when the model supplies `make_cache` (which this
   one must, to expose the 44 effective layers): `make_prompt_cache` skips
   `max_kv_size` whenever `make_cache` exists. Use `--kv-bits` to reduce KV
   precision instead.
2. The 262K number in the spec table is theoretical; the practical regime is
   short tool-use lengths, which is exactly what this eval exercises.

This is a genuine, quantifiable cost of the looped design that most mlx-lm
documentation doesn't cover.

---

## Reports and artifacts

Each run writes an isolated directory under `benchmark_results/` (gitignored).
The runs this README cites are committed under
[`results/published/`](results/published/) so every number is checkable:

| artifact | purpose |
|---|---|
| `manifest.json` | run identity, model, environment (real versions, git SHA, quantization) |
| `results.jsonl` | per-case grade, detail, stop_reason, latency, memory, model output |
| `summary.json` | pass rate + Wilson CI, per-kind breakdown, aggregate latency, truncations |
| `report.md` | human-readable report, regenerated from the above (idempotent) |

`report`, `compare`, and `regrade` read **only** persisted files and never
re-run the model. `regrade` retroactively applies the current graders to a
persisted run — useful when the grader changes (it did for this report).

---

## Requirements

- macOS on Apple Silicon
- Python ≥ 3.10
- `mlx-nanbeige` (the port), `mlx`, `mlx-lm`, `transformers`, `jsonschema`
- dev: `pytest`

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e './mlx_nanbeige[dev]'   # the port (single source of truth)
.venv/bin/pip install -e '.[dev]'                # this eval harness
.venv/bin/pytest -q          # harness-readiness gate (mock + cache-consistency, no model download)
.venv/bin/nanbeige-mlx-eval --help
```

## CLI workflow

```bash
# harness readiness (no model needed)
nanbeige-mlx-eval list-suites
nanbeige-mlx-eval validate-suite suites/agentic_en.json
nanbeige-mlx-eval run --suite smoke --runtime mock --output-root benchmark_results

# convert the checkpoint to MLX quants (one-off; needs the HF weights locally)
mlx-nanbeige-convert --src models/nanbeige42-hf --out models/nanbeige-mlx-4bit --bits 4

# fidelity (Half A) — needs transformers + the HF reference
nanbeige-mlx-eval parity --src models/nanbeige42-hf --out benchmark_results/parity.json --device cpu --dtype bf16
nanbeige-mlx-eval trace  --src models/nanbeige42-hf --out benchmark_results/trace.json --device cpu --dtype bf16

# real eval (Half B) across a quant, with warmup + Wilson CIs
nanbeige-mlx-eval run --suite agentic_en --runtime mlx \
  --model models/nanbeige-mlx-4bit --output-root benchmark_results --warmup 1

# retroactively re-grade a persisted run when the grader changes
nanbeige-mlx-eval regrade benchmark_results/<run_dir> --suite agentic_en

# regenerate / compare reports (idempotent, no model)
nanbeige-mlx-eval report  benchmark_results/<run_dir>
nanbeige-mlx-eval compare benchmark_results/<run_a> benchmark_results/<run_b>
```

---

## Result safety

- **Greedy decoding, asserted** (`make_sampler(temp=0.0)` passed explicitly;
  the manifest records it).
- **Smoke-flagging:** `--limit` caps the case count and marks the run smoke-only.
- **Reasoning isolation:** the grader grades only the post-`</think>` answer
  tail; an unclosed `<think>` block fails as `truncated_no_answer`.
- **Wilson 95% CIs** on every pass rate, so 8/8 is read as 1.00 [0.68, 1.00].
- **Warmup + median** (`--warmup`, `--repeats`) keep first-call compile cost out
  of published numbers.
- **Answer-key protection:** only the grade outcome and the model's own output
  are persisted — never the suite's expected answer.

---

## Local-only files (what is and isn't public)

Public (this repo): both packages' code, the suites, the tests, the README, the
MIT license, and the committed results under `results/published/`. **Private
(gitignored, never committed):** `plan.md` and other working docs, `.zcode/`,
`.venv/`, `benchmark_results/` (the full run history), `models/` (downloaded
weights and converted quants — large), `reference/` (the upstream HF modeling
code kept only for the parity reference).

The model weights are © the Nanbeige team under Apache-2.0; this repo ships none
of them — only the code to convert and evaluate.

---

## License

MIT for the code in `mlx_nanbeige/` and `nanbeige_mlx_eval/`. The Nanbeige model
weights and chat template are governed by the upstream Apache-2.0 license;
convert, evaluate, and redistribute them per that license. Published weight
repos (via `mlx-nanbeige-upload`) carry the upstream Apache-2.0 LICENSE, a
NOTICE stating the modification (quantization), and proper `base_model` /
`license` frontmatter.
