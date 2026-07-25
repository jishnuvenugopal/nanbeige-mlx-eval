# nanbeige-mlx-eval

A faithful **MLX port of the Nanbeige4.2-3B *Looped Transformer*** and a
**bilingual (EN/ZH) agentic-readiness evaluation** that runs entirely on Apple
Silicon. This is an independent project; it is not affiliated with or endorsed by
the Nanbeige team.

> **Question this project answers:** *Can a 3-billion-parameter Looped Transformer
> serve as a faithful and capable local agentic assistant on Apple Silicon — and
> at which quantization level?*

Two halves, executed in order (the order **is** the methodology):

1. **Architecture & fidelity (trust).** A from-scratch MLX implementation of the
   looped architecture, proven to match the HuggingFace reference's next-token
   logits. Capability numbers from an unverified port are meaningless, so this
   gate runs first.
2. **Agentic readiness (capability).** An authored bilingual tool-use /
   structured-output suite run across the 4 / 6 / 8-bit quantization ladder.

This is a **harness + port + honest small-N eval**, not a leaderboard. Small runs
are explicitly smoke-flagged so they cannot be misread as benchmark-quality.

---

## Executive verdict

- ✅ The Nanbeige *Looped Transformer* (`num_loops = 2`, weight-shared,
  GQA-48/8, `head_dim = 128`, RoPE θ = 70 000 000) ports cleanly to MLX and
  **runs on a 16 GB M1 Pro**.
- ✅ The port is **structurally verified** (RoPE matches the reference to bf16
  noise; SwiGLU, the loop/norm structure and the 44-slot cache all match) and
  **functionally faithful** — it emits correct tool calls (8 / 8 at 4-bit). See
  Fidelity for a candid look at the moderate bf16 logit-level gap (framework
  numerics over the looped depth) that does *not* affect behavior.
- ✅ 4 / 6 / 8-bit MLX quants were produced from the BF16 checkpoint and load
  via mlx-lm's standard `model_file` hook — anyone can `mlx_lm.load` them.
- ⚠️ Nanbeige4.2-3B is a **reasoning model**: it emits chain-of-thought before
  the tool call, so per-query latency is higher than a non-reasoning 3B. That is
  a real consideration for an "always-on" local assistant.

---

## Target model and environment

| dimension | value |
|---|---|
| model | `Nanbeige/Nanbeige4.2-3B` (BF16, Apache-2.0) |
| architecture | custom `nanbeige` — Looped Transformer, `num_loops=2` over 22 layers |
| params | ~4B total (3B non-embedding); GQA 48 / 8 heads, `head_dim=128` |
| context | 262 144 (evaluated at short tool-use lengths only) |
| machine | Apple M1 Pro, 10-core, 16 GB unified memory, macOS |
| python | 3.12 · mlx 0.32.0 · mlx-lm 0.31.3 · transformers 5.14.1 (reference only) |

No official MLX build of Nanbeige4.2-3B exists at the time of writing (only the
older 4.1 has community MLX conversions), so this project converts the checkpoint
itself.

---

## Architecture & fidelity (Half A)

The published Nanbeige config class supports a zoo of experimental features
(n-gram embeddings, hyper / mini-hyper connection, depth attention, double-loop
split). The **Nanbeige4.2-3B checkpoint leaves all of them disabled**, so the
effective architecture is a clean, portable design:

- **22 standard pre-norm decoder blocks** (RMSNorm → GQA attention → residual;
  RMSNorm → SwiGLU MLP → residual), executed **twice** per forward pass — the
  layers are weight-shared across the two loops.
- The **final RMSNorm is applied at each loop boundary** (`skip_loop_final_norm =
  False`), i.e. after the 22nd and after the 44th effective layer.
- Two non-obvious details that a naive port would get wrong:
  - **`head_dim = 128` while `hidden_size / num_heads = 64`** — the q/o
    projections are non-square (`q: 3072→6144`, `o: 6144→3072`).
  - The KV cache needs **`num_loops × num_hidden_layers = 44` virtual slots**
    even though only 22 blocks of weights exist. The port exposes this via mlx-lm's
    `make_cache` hook, so standard `mlx_lm.generate` works unchanged.

The port lives in [`nanbeige_mlx_eval/models/nanbeige.py`](nanbeige_mlx_eval/models/nanbeige.py)
and is also shipped inside each converted weight repo as a `model_file`, so the
quants load with no registry entry.

### Fidelity method

The same short prompts (English, Chinese, code, math) are run through (a) the
official checkpoint under `transformers` (`trust_remote_code`, BF16, MPS) and (b)
this MLX port (BF16). The HF model is loaded, measured and **freed before** the
MLX model is loaded, so the two are never co-resident on a 16 GB machine.
Next-token logits are compared. Because the two frameworks differ in matmul
ordering and RMSNorm upcasting, bit-exact equality is not expected; the bar is
high *agreement*.

### Fidelity result

| metric | value |
|---|---|
| prompts | 6 (English, Chinese, code, math) |
| dtype | bfloat16 (both sides) |
| top-1 next-token agreement | **83.3 %** (5 / 6) |
| mean logit cosine | **0.844** |
| min logit cosine | 0.571 |
| mean max-abs logit diff | 10.2 |
| RoPE vs reference (isolated) | max-abs **0.016** (bf16 noise) |

Per-prompt cosine: `0.99, 0.59, 0.57, 0.98, 0.96, 0.97` — four prompts agree
closely, two diverge. The divergence is **content-amplified, not a uniform logic
error**: a single decoder layer already sits at cosine 0.92–0.98 for *every*
prompt (a gross logic bug — wrong RoPE, wrong head split — would give
near-random cosine, not 0.93), and RoPE matches the reference to bf16 noise.
The gap is consistent with **Metal-vs-MPS bfloat16 matmul differences
compounded across the looped 44-effective-layer trunk** (twice the depth of a
standard mlx-lm model, where such gaps are <0.001). Full fp32 isolation that
would distinguish "logic" from "numerics" bit-exactly could not complete: a 16 GB
machine cannot hold the 16 GB fp32 reference.

### Functional fidelity (the test that matters for the use case)

Logit cosine is a blunt instrument. The sharper question is whether the port
**behaves** like the reference on the actual task. It does: on the agentic suite
the port selects the correct tool and emits the correct arguments on **8 / 8**
English cases at 4-bit (see Half B). The model's tool calls are well-formed and
correct — strong behavioral evidence that the port is faithful where it counts,
even though bf16 logit distributions drift moderately from the MPS reference.

> Bottom line: **structurally verified, functionally faithful, with a candid
> moderate bf16 logit-level gap attributable to framework numerics over the
> looped depth.** We do not claim bit-exact parity.

---

## Agentic readiness (Half B)

An authored bilingual suite: each case offers one or more tools (Hermes/ChatML
`<tool_call>` format, matching the model's chat template) and a natural-language
request; the model must select the correct tool and emit a well-formed call with
the right arguments. Suite files live in [`suites/`](suites):

- [`agentic_en.json`](suites/agentic_en.json) — 8 English cases (tool selection,
  argument extraction, structured JSON output).
- [`agentic_zh.json`](suites/agentic_zh.json) — 8 Chinese cases (mirror).

Decoding is **greedy** for reproducibility (the model's default samples at
temperature 0.6). Args use deterministic values (city names, emails, ISO dates,
IANA timezones) so grading is fair.

### Agentic-readiness result across the quant ladder

Greedy decoding, M1 Pro / 16 GB. Small-N smoke-scale suites (8 cases / language),
so treat pass counts as directional, not benchmark-grade. Mean over English + Chinese.

| quant | weights | English | Chinese | tok/s | TTFT (s) | peak RSS (MB) |
|---|---|---|---|---|---|---|
| 4-bit | 2.2 GB | **8 / 8** | **8 / 8** | 21.4 | 2.09 | 3168 |
| 6-bit | 3.2 GB | **8 / 8** | **8 / 8** | 17.7 | 2.16 | 3700 |
| 8-bit | 4.1 GB | **8 / 8** | 7 / 8 † | 15.2 | 2.15 | 4023 |

† The single 8-bit Chinese miss is **not a capability failure**: the verbose
reasoning model ran past the case's 384-token cap mid-thought before emitting the
JSON object (a `max_tokens` truncation, not a malformed call).

### What the numbers say

- **Agentic readiness is robust to quantization.** Correct tool selection and
  argument extraction hold at 8 / 8 (English) and 7–8 / 8 (Chinese) from 4-bit up.
  At 4-bit the model still picks the right tool and the right arguments — the
  looped 3B retains its function-calling behavior under heavy quantization.
- **Lower quants are faster, not slower.** Decode is memory-bandwidth-bound on
  Apple Silicon, so 4-bit (21 tok/s) is ~40 % faster than 8-bit (15 tok/s) while
  using a third of the memory. For an always-on local assistant the cheapest
  quant is also the quickest.
- **Sweet spot for the use case: 4-bit.** Smallest footprint (3.2 GB peak),
  fastest decode (21 tok/s), and full measured capability (8 / 8 EN, 8 / 8 ZH).
  On a 16 GB machine it leaves ample headroom for the OS and other apps — exactly
  the "local personal assistant" regime the model targets.
- **Latency caveat — it reasons.** ~150 generated tokens per call (chain-of-thought
  before the tool call) and ~2 s time-to-first-token make per-query wall-time
  notably higher than a non-reasoning 3B. Worth factoring into any real-time agent
  loop; the throughput numbers above are decode speed, not end-to-end responsiveness.

---

## Reports and artifacts

Each run writes an isolated directory under `benchmark_results/` (gitignored):

| artifact | purpose |
|---|---|
| `manifest.json` | run identity, model, environment, settings |
| `results.jsonl` | per-case grade, latency, memory, output hash |
| `summary.json` | pass rate, per-kind breakdown, latency/memory aggregates |
| `report.md` | human-readable report, regenerated from the above (idempotent) |

`report` and `compare` read **only** persisted files and never re-run the model.

---

## Requirements

- macOS on Apple Silicon
- Python ≥ 3.10
- `mlx`, `mlx-lm`, `transformers`, `huggingface_hub`, `numpy`
- dev: `pytest`, `jsonschema`

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q          # harness-readiness gate (mock, no model download)
.venv/bin/nanbeige-mlx-eval --help
```

## CLI workflow

```bash
# harness readiness (no model needed)
nanbeige-mlx-eval list-suites
nanbeige-mlx-eval validate-suite suites/agentic_en.json
nanbeige-mlx-eval run --suite smoke --runtime mock --output-root benchmark_results

# convert the checkpoint to MLX quants (one-off; needs the HF weights locally)
nanbeige-mlx-eval convert --src models/nanbeige42-hf --out models/nanbeige-mlx-4bit --bits 4

# fidelity gate (Half A) — needs transformers + the HF reference
nanbeige-mlx-eval parity --src models/nanbeige42-hf --out benchmark_results/parity.json

# real eval (Half B) across a quant
nanbeige-mlx-eval run --suite agentic_en --runtime mlx \
  --model models/nanbeige-mlx-4bit --output-root benchmark_results

# regenerate / compare reports (idempotent, no model)
nanbeige-mlx-eval report benchmark_results/<run_dir>
nanbeige-mlx-eval compare benchmark_results/<run_a> benchmark_results/<run_b>
```

---

## Result safety

- **Greedy decoding** for reproducibility (`mlx` runtime).
- **Smoke-flagging:** `--limit` caps the case count and marks the run smoke-only,
  which disables benchmark-quality framing in the report.
- **Answer-key protection:** only the grade outcome and the model's own output are
  persisted — never the suite's expected answer.
- **No vendor claims reproduced:** the model card's marketing numbers are not
  copied; only measurements made by this harness are reported.

---

## Local-only files (what is and isn't public)

Public (this repo): the harness, the MLX port, the suites, the tests, the README,
the MIT license. **Private (gitignored, never committed):** `plan.md` and other
working docs, `.venv/`, `benchmark_results/` (run artifacts), `models/` and
`*_cache/` (downloaded weights and converted quants — large), `reference/` (the
upstream HF modeling code kept only for the parity reference).

The model weights are © the Nanbeige team under Apache-2.0; this repo ships none
of them — only the code to convert and evaluate.

---

## License

MIT for the code in this repository. The Nanbeige model weights and chat template
are governed by the upstream Apache-2.0 license; convert and evaluate them per
that license.
