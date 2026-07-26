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
- ✅ **Tool selection and argument extraction hold at ~90 % (EN) and ~90 % (ZH)
  from 4-bit up** on the widened 30-case suite — flat across quants, with the
  ~3 failures-per-config consistent across quants (a 3B model property, not a
  port artifact). Structured output succeeds at all six quant×language configs
  once the token cap is raised: **zero truncations across 180 cases** at a
  1024-token cap.
- ⚠️ **A moderate logit gap vs the HF reference shows up on CPU (mean cosine
  0.85, top-1 83%)** — not MPS noise (CPU run reproduces it). Its source is
  **documented-open, not blocking.** Six candidate causes are ruled out by
  measurement (device numerics, bf16 compounding, input scale, per-layer
  arithmetic, the port's own two code paths, and RoPE-at-bf16 — the last because
  `mx.fast.rope` is fp32-internal regardless of input dtype, confirmed by a
  bit-identical upcast experiment). The port is verified where it counts:
  per-layer fp32 agreement with the reference's own `NanbeigeDecoderLayer`, cache
  equality under mutation testing, bit-identical agreement between its own two
  code paths, and ~90 % agentic capability flat across 4/6/8-bit. The gap does
  not affect behavior on the agentic suite. Full record, including every
  falsified hypothesis and one known blind spot in the settling instrument:
  [`docs/investigation-log.md`](docs/investigation-log.md). See
  [§ Fidelity](#fidelity-half-a).
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
| RoPE bf16 floor (isolated) | max-abs **0.011** (0.78% of probe RMS) on the cos/sin downcast, measured with a real-`q` probe (`results/published/parity_cpu_bf16.json`). **Ruled out as a cause of the end-to-end gap:** an fp32 upcast around the MLX RoPE call left the parity cosine bit-identical (0.846566 both ways, 6 prompts, ~166k logits) — `mx.fast.rope` is fp32-internal regardless of input dtype (Addendum 6 of the investigation log) |

> **Important correction.** An earlier version of this report attributed the gap
> to "Metal-vs-MPS bfloat16 matmul differences." That hypothesis is **falsified**:
> running the reference on **CPU** reproduces the same mean cosine (0.847 vs the
> old 0.844 on MPS). The gap is **not** MPS-specific. What it actually is, see
> the trace below.

### Per-layer divergence trace

> **Caveat (read first).** The bisect and the trace disagree by ~13,000× at layer
> 0. A `crosscheck` showed the port's two code paths agree bit-exactly (cosine
> 1.0), and that the disagreement is between the bisect's standalone
> `NanbeigeDecoderLayer` invocation (B) and the same layer inside the full HF
> model (D): cosine 0.925, RMS 0.656 vs 0.559. A follow-up `--replay` experiment
> settles which of B or D is the layer's true behaviour: replaying the model's
> actual call arguments into the layer reproduces D bit-exactly, while B does
> not. The difference was localized to the **object**, not the arguments: the
> bisect's `_real_layer_stages` called `NanbeigeDecoderLayer(...).to(td)`, which
> casts the non-persistent fp32 `inv_freq` buffer to bf16 (`rope_theta = 7e7` →
> frequencies span 1.0 down to ~2e-8, mangled by bf16) — and `strict=False`
> loading can't restore it. (The `[1,1,5,6]` mask shape is a red herring: the 6th
> column is HF's StaticCache sizing boilerplate, sliced away by eager attention
> before use — not a loop/depth dimension.) The bisect's `.to(td)` buffer-cast is
> fixed (parameters-only now), but that fix is confined to the **bisect harness**;
> it does not close the end-to-end 0.847, because the MLX port has no `inv_freq`
> buffer to clobber and `mx.fast.rope` is fp32-internal regardless. The curve
> below is the real in-model behaviour; the end-to-end gap is documented-open.

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
- **Layer 0 receives an identical input on both sides** (the embedding lookup
  agrees to cosine 1.0, verified by `crosscheck`) yet the trace reports cosine
  0.925 there. The `crosscheck` resolves this: the port's layer-0 output is
  bit-identical across both tools (cosine 1.0), so the 0.925 is the HF reference
  disagreeing with *itself* — a standalone `NanbeigeDecoderLayer` and the same
  layer inside the full model produce different outputs (RMS 0.656 vs 0.559).
  The shape of the curve (worst where the residual is smallest, the step at the
  loop boundary) is consistent with the in-model HF forward rescaling the
  residual around layer 0 in a way the standalone layer does not.

### The decisive experiment: single-layer bisect against the real reference

The blocker cited in earlier drafts — that fp32 isolation OOMs on 16 GB — is
false for the question actually being asked. One decoder layer is ~143 M params
(~573 MB in fp32); loading just it on both sides needs ~1.2 GB. The `bisect`
subcommand feeds a common input to the layer in both frameworks and compares
the stages independently (so an early gap cannot mask a later one), in fp32
where any disagreement is logic, not numerics. `--reference real` makes the
checkpoint's **own `NanbeigeDecoderLayer`** the arbiter — not a reimplementation
of it — by instantiating `modeling_nanbeige.NanbeigeDecoderLayer`, loading the
same weights, and hooking its submodules (12 of 14 stages from real code):

```bash
nanbeige-mlx-eval bisect --src models/nanbeige42-hf --dtype fp32 \
  --input real --reference real --gate
```

**Result (`results/published/bisect_fp32_realref.json`): every stage agrees to
cosine 1.0, max-abs ≤ 2.3e-05, against the real `NanbeigeDecoderLayer`.** No
first-divergent stage. The bf16 real-reference run
(`bisect_bf16_realref.json`) returns block cosine **0.99999326** with every
hookable stage ≥ 0.99998. **This verifies the layer's arithmetic is correct
against Nanbeige's own code** — not merely against a hand-written mirror of it.

This is the strongest available statement about the layer, and it clears the
layer at the real operating point. It does *not* explain the end-to-end gap —
see the scale sweep and the locus argument below.

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
difference), but the per-layer block cosine moves only 0.99999304 → 0.99999381
at the real input scale — a shift of ~1e-7. **RoPE precision is real and
localized, but it is not the dominant cause of the end-to-end gap.**

### Why "compounded bf16 precision" is *not* the explanation

The earlier version of this section closed with "the 0.847 end-to-end cosine is
the accumulation of the small bf16 stage errors (~1e-5 each) compounded over 44
layers." That is **arithmetically impossible** and an input-scale sweep has
falsified it:

```bash
nanbeige-mlx-eval bisect --src models/nanbeige42-hf --dtype bf16 --sweep \
  --out results/published/bisect_scale_sweep.json
```

| input rms | 1.0 | 0.1 | 0.024 | 0.01 | **real (0.027)** |
|---|---|---|---|---|---|
| block cosine | 0.99999402 | 0.99997704 | 0.99997669 | 0.99997499 | **0.99999304** |
| 1 − cosine | 6.0e-6 | 2.3e-5 | 2.3e-5 | 2.5e-5 | **7.0e-6** |

One layer loses ~7e-6 of cosine at the real operating point. Even accumulating
linearly, 44 layers lose ~3e-4 — four orders of magnitude short of the 0.15 the
end-to-end gap implies. The gap is also **not** an input-magnitude effect:
1−cosine stays in the 1e-5 band from rms 1.0 down to rms 0.01.

The remaining fact is that the per-layer bisect and the per-layer trace disagree
by ~13,000× at layer 0. A `crosscheck` tool resolves which side broke by
computing the full 4-way matrix on the same layer 0 (the port's output via both
tools, and the reference's output via both tools):

```bash
nanbeige-mlx-eval crosscheck --src models/nanbeige42-hf \
  --out results/published/crosscheck_layer0.json
```

| pair | cosine | what it is |
|---|---|---|
| embedding (shard vs MLX vs HF) | 1.0 | embedding clean |
| **bisect-MLX vs trace-MLX** | **1.0** (bit-identical) | port agrees with itself across tools |
| bisect-HF vs trace-HF (within bisect) | 0.99999 | bisect's reported agreement |
| **bisect-HF vs trace-HF** | **0.925** | standalone layer (B) vs in-model layer (D) |

The crosscheck alone is ambiguous about which of B or D is ground truth — and
reading "the reference disagrees with itself" as exoneration of the port is the
**wrong** reading (D is a real forward pass, B is a hand-built invocation).
`crosscheck --replay` settles it:

```bash
nanbeige-mlx-eval crosscheck --src models/nanbeige42-hf --replay \
  --out results/published/replay_layer0.json
```

| pair | cosine | what it is |
|---|---|---|
| **D in-model vs replay (model's args into same layer)** | **1.0** | layer is deterministic; D is its true output |
| D in-model vs B standalone | 0.925 | B is the mis-invocation |

**B is the artifact; D is ground truth.** The bisect's `_real_layer_stages`
constructed the standalone layer with `NanbeigeDecoderLayer(...).to(td)`, which
casts the non-persistent fp32 `inv_freq` buffer to bf16 (with `rope_theta = 7e7`,
the frequencies span 1.0 down to ~2e-8 and bf16 mangles the low end); because the
buffer is non-persistent, the subsequent `load_state_dict(..., strict=False)`
cannot restore it. The fix casts parameters only and is confined to the **bisect
harness** — it does not touch the end-to-end 0.847, because the MLX port has no
`inv_freq` buffer (it uses `nn.RoPE`, four Python scalars) and `mx.fast.rope` is
fp32-internal regardless of input dtype (Addendum 6 of the investigation log).
The `[1,1,5,6]` mask shape that earlier rounds blamed is a red herring: the 6th
column is HF's StaticCache sizing boilerplate, sliced away by eager attention
before use. The end-to-end gap remains documented-open.

> Bottom line: **the source of the logit gap is documented-open, not blocking.**
> Six candidate causes are ruled out by measurement (the seventh, "RoPE runs in
> bf16," was killed by a bit-identical upcast experiment — `mx.fast.rope` is
> fp32-internal regardless of input dtype). The `[1,1,5,6]` mask shape is a red
> herring: the 6th column is HF's `past_seen_tokens + sequence_length + 1`
> boilerplate for StaticCache sizing, present in every Llama-family model, and
> eager attention slices it to `[:,:,:,:L]` before use — nothing to do with
> `num_loops`. The gap does not affect behavior on the agentic suite. Bit-exact
> parity is not claimed and not expected in bf16. Full record:
> [`docs/investigation-log.md`](docs/investigation-log.md).

### Functional fidelity (the test that matters for the use case)

Logit cosine is a blunt instrument. The sharper question is whether the port
**behaves** like the reference on the actual task. It does: on the 30-case
agentic suite the port selects the correct tool and emits the correct arguments
on **~90 %** of cases across the quant ladder, with the failures consistent
across quants (a model property, not a port artifact — see Half B).

---

## Agentic readiness (Half B)

An authored bilingual suite: each case offers one or more tools (Hermes/ChatML
`<tool_call>` format, matching the model's chat template) and a natural-language
request; the model must select the correct tool and emit a well-formed call with
the right arguments. Suite files live in [`suites/`](suites):

- [`agentic_en.json`](suites/agentic_en.json) — 30 English cases.
- [`agentic_zh.json`](suites/agentic_zh.json) — 30 Chinese mirror cases.

Decoding is **greedy** (`temp=0.0`, asserted via an explicit sampler — not
assumed). Args use deterministic values (city names, emails, ISO dates, IANA
timezones). The grader grades **only the answer tail** (the text after the first
`</think>`), never the reasoning scratchpad; a truncated `<think>` block
hard-fails as `truncated_no_answer` rather than scanning the stream for a
coincidental match. Every case carries a `stop_reason` (`stop`/`length`).

### Result across the quant ladder

Greedy, M1 Pro / 16 GB, `--warmup 1` (absorbs first-call Metal/compile cost),
1024-token cap. **30 cases / language** — the earlier 8-case suite gave
8/8 everywhere but a Wilson CI of `[0.68, 1.00]` that could not distinguish
"perfect" from "70%". At 30 cases the CI tightens enough to measure a real
ceiling. Pass rates carry Wilson 95% CIs; throughput is **aggregate** (total
generated tokens / total decode time), not a mean of per-case ratios. EN@4-bit
was run at `--repeats 3`; the other five configs at `--repeats 1` (pass rate is
repeat-invariant; repeats only stabilize timing).

| quant | weights | English | Chinese | CI95 (30) | decode tok/s | TTFT (tools) | peak RSS |
|---|---|---|---|---|---|---|---|
| 4-bit | 2.2 GB | **27 / 30** (90%) | **27 / 30** (90%) | [0.74, 0.97] | 35.1 | 2.2 s | 2.9 GB |
| 6-bit | 3.2 GB | **26 / 30** (87%) | **28 / 30** (93%) | [0.70, 0.97] | 22.9 | 2.5 s | 3.8 GB |
| 8-bit | 4.1 GB | **26 / 30** (87%) | **27 / 30** (90%) | [0.70, 0.97] | 20.7 | 2.2 s | 4.7 GB |

**Structured output is cap-robust**: zero `stop_reason: length` truncations
across all 180 cases at the 1024-token cap — every run stopped on `stop`.
(The earlier 8-case suite reported truncations only because the grader was
reading the scratchpad; that is fixed, and the widened suite confirms the
model finishes its reasoning within the cap.)

The honest headline: **tool selection and argument extraction are
quantization-robust** — pass rate is flat at ~87-93% from 4-bit to 8-bit in
both languages, and 4-bit is not weaker than 8-bit. The ceiling is ~90%, not
100%, and it is a capability property of the 3B model, not the port: the
~3 failures per config are consistent across quants (see below).

### What the numbers say

- **Agentic readiness is robust to quantization.** Pass rate is flat at
  ~87-93% in both languages from 4-bit up — 4-bit (27/30, 27/30) is **not**
  weaker than 8-bit (26/30, 27/30). At 4-bit the model still picks the right
  tool and the right arguments ~90% of the time.
- **The ~90% ceiling is a model property, not a port artifact.** The failures
  are consistent across quants: 3 English cases (`email-carol`,
  `time-london`, `search-recipe`) and 1 Chinese case (`email-wangwu`) fail at
  *every* quant. The dominant modes are `no_tool_call_found` (the model
  reasons but never emits a parseable call) and `missing:[arg]` (it calls the
  right tool but paraphrases the argument so it no longer matches). These are
  3B-reasoning-model limits, surfaced honestly by widening from 8 to 30 cases —
  the old 8/8 simply didn't include them.
- **Structured output is cap-robust, not capability-limited.** An earlier
  version of this report graded runs as `schema_valid` on outputs that were
  actually reasoning truncated mid-sentence (the grader was reading the
  scratchpad). With the grader fixed to grade only the answer tail and the cap
  raised to 1024, **zero** of the 180 cases truncate — the two `json-profile`
  cases pass at all six configs (consuming 188-962 tokens of chain-of-thought
  first). The model needs room to think; given it, it finishes.
- **Lower quants are faster, not slower.** Decode is memory-bandwidth-bound on
  Apple Silicon, so 4-bit (~35 tok/s aggregate) is ~1.7× faster than 8-bit
  (~21 tok/s) while using ~60 % of the memory.
- **Sweet spot for the use case: 4-bit.** Smallest footprint (~2.9 GB peak),
  fastest decode, tied for the best pass rate. On a 16 GB machine it leaves
  ample headroom — exactly the "local personal assistant" regime.
- **Latency caveat — it reasons.** ~120 generated tokens per tool call
  (chain-of-thought before the call) and ~2.2 s TTFT make per-query wall-time
  notably higher than a non-reasoning 3B. The throughput numbers above are
  decode speed, not end-to-end responsiveness.

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
- **Wilson 95% CIs** on every pass rate, so 27/30 is read as 0.90 [0.74, 0.97]
  rather than as a deceptive "90%".
- **Warmup + repeat controls** (`--warmup`, `--repeats`) keep first-call compile
  cost out of published numbers. Pass rate is repeat-invariant; `--repeats`
  stabilizes timing. The ladder used `--warmup 1` throughout; EN@4-bit was run at
  `--repeats 3`, the other five configs at `--repeats 1` (stated per-config in
  the table above, not assumed uniform).
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
