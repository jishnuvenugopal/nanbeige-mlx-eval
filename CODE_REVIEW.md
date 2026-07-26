# Review: `nanbeige-mlx-eval` + `mlx-nanbeige`

Reviewed 2026-07-25 against `reference/modeling_nanbeige.py`, `configuration_nanbeige.py`,
the committed `benchmark_results/`, and the installed mlx 0.32.0 / mlx-lm 0.31.3.

Verdict up front: **the port looks architecturally correct** — I could not find a
logic bug. But two claims in the READMEs don't survive contact with the artifacts:
the fidelity number is very likely a measurement artifact you've attributed to the
wrong cause, and the "8/8 across the quant ladder" table contains four grader false
positives. Fix those two and this is a genuinely strong project.

---

## What I verified as correct

Line-by-line against the reference, for this checkpoint:

| detail | reference | port | |
|---|---|---|---|
| eager attention scale | `/ math.sqrt(head_dim)` (128) | `head_dim ** -0.5` | ✓ |
| RoPE convention | `rotate_half`, `cat((freqs,freqs))` | `traditional=False` | ✓ |
| RoPE dims | `NanbeigeRotaryEmbedding(head_dim)` = 128 | `initialize_rope(head_dim)` | ✓ |
| SwiGLU order | `act_fn(gate) * up` | `swiglu(gate, up)` = `silu(gate)*x` | ✓ |
| GQA head mapping | `repeat_kv`, q head → `h // n_rep` | mlx sdpa, same grouping | ✓ |
| loop KV slot index | `layer_idx + loop_idx * n_layers` | `loop * n_layers + i` | ✓ |
| final norm placement | each loop boundary (`skip_loop_final_norm=False`) | same | ✓ |
| qk_layernorm | default `False`; no q/k norm tensors in the index | absent | ✓ |
| weight key mapping | 201 tensors, 12 unique patterns | 1:1 with module tree | ✓ |
| disabled feature zoo | ngram / hyper-connection / MHC / depth-attn / double-loop all off | not implemented | ✓ |

The 44-slot `make_cache` hook is the right call and `make_prompt_cache` does defer
to it (verified in the installed source). The `model_file` approach is clean.

---

## Blockers

### A1. The fidelity explanation is self-refuting

The README says the divergence is *"Metal-vs-MPS bfloat16 matmul differences
compounded across the looped 44-effective-layer trunk."* But it also says:

> a single decoder layer already sits at cosine 0.92–0.98 for *every* prompt

A single bf16 decoder layer should agree to cosine **> 0.9999**, not 0.92. After one
layer there is nothing to compound — so "compounded across 44 layers" cannot be the
explanation for a gap that is already present at layer 1. That one-layer number is the
finding, and it's being read as reassurance when it's the smoking gun.

Since I can't find an architectural bug, the likeliest culprit is the **reference side**:

1. **PyTorch MPS bf16.** `parity.py` runs the reference on `.to("mps")` in bfloat16.
   MPS bf16 accuracy is known-shaky. This is the single most likely cause and the
   cheapest to rule out.
2. **The reference downcasts RoPE cos/sin to bf16** before applying them
   (`return cos.to(dtype=x.dtype)`, line ~964), while `mx.fast.rope` computes and
   applies in fp32. With θ = 70,000,000 and `head_dim` 128, almost every frequency
   sits where `cos ≈ 1`, and bf16 spacing near 1.0 is 2⁻⁸ ≈ 0.4%. So the port is
   *more* accurate than the reference here, and any strict logit comparison has a
   floor built into it. Note the interesting corollary: your `max-abs 0.016` "RoPE
   matches to bf16 noise" result is consistent with the reference being the noisy one.

**Do this:**

- Rerun parity with the reference on **`cpu`**, bf16. 4B × 2 bytes ≈ 8.3 GB — fits in
  16 GB, and you already free the HF model before loading MLX. If cosine jumps to
  > 0.99, you're done and the README gets a much better headline.
- Add a fp32 spot-check on a **single layer** on CPU (one layer is ~200 MB in fp32).
  That isolates logic from numerics without needing the 16 GB full-fp32 model you
  currently cite as the blocker.
- Until then, don't ship the "**Verified fidelity**" heading with a mean cosine of
  0.844 and a min of 0.571 under it. The `parity.py` docstring already says the bar
  is "logit cosine ≈ 1"; the result doesn't meet the bar the code sets for itself.

### A2. Missing the two tests that would establish trust — both torch-free

`test_make_cache_slot_count` asserts `len(cache) == 44`. That's a list length, not
behavior. Nothing in either repo tests a single number the model produces.

- **Prefill-vs-decode self-consistency.** Generate N tokens with the cache, then
  re-run the full sequence as one prefill pass and compare logits. This catches every
  offset/slot bug in the 44-slot scheme and needs no reference model. For a
  weight-shared looped architecture where two virtual slots per layer must stay in
  lockstep, this is *the* test. Right now nothing would catch a swapped loop index.
- **Per-layer / per-loop divergence trace, as a CLI command.** Dump hidden states
  after each of the 44 effective layers on both sides and plot cosine vs depth.
  Gradual decay = numerics; a step = a bug. You clearly ran this ad hoc to get the
  0.92–0.98 figure — make it a first-class command so the answer to "is it a bug?" is
  a chart, not a paragraph.

### A3. `o_proj` hardcodes `bias=False`

```python
self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)   # port
self.o_proj = nn.Linear(..., bias=config.attention_bias)        # reference
```

Harmless here (`attention_bias: false`), but this file is embedded as `model_file`
inside every weight repo you publish. Use `args.attention_bias`.

---

## Eval integrity

### B1. The grader is reading the model's scratchpad — 4 of 6 committed runs have a false pass

The chat template always appends `<think>\n`, so every generation opens with a
reasoning block. `_extract_first_json_object` scans the **whole** output, including
that block. Result, straight from `benchmark_results/`:

| quant | case | gen tokens | hit 384 cap | closed `</think>` | graded |
|---|---|---|---|---|---|
| 4-bit | `en-json-profile` | 188 | no | yes | pass ✓ |
| 6-bit | `en-json-profile` | 384 | **yes** | **no** | pass ✗ |
| 8-bit | `en-json-profile` | 384 | **yes** | **no** | pass ✗ |
| 4-bit | `zh-json-profile` | 384 | **yes** | **no** | pass ✗ |
| 6-bit | `zh-json-profile` | 384 | **yes** | **no** | pass ✗ |
| 8-bit | `zh-json-profile` | 384 | **yes** | **no** | fail ✓ |

The 8-bit EN output ends mid-sentence — *"Let me write it"* — having never closed its
think block or emitted an answer. It scored `schema_valid`, because the model happened
to write a conforming JSON object while reasoning about JSON.

Corrected table:

| quant | English | Chinese | README claims |
|---|---|---|---|
| 4-bit | 8/8 | **7/8** | 8/8, 8/8 |
| 6-bit | **7/8** | **7/8** | 8/8, 8/8 |
| 8-bit | **7/8** | **7/8** | 8/8, 7/8 |

The README explains the 8-bit ZH miss as a `max_tokens` truncation — correct, but the
*same* truncation happened in four other runs and was silently scored as a pass. The
real finding is more interesting than the one you published: **structured output is
the one thing that's fragile at every quant**, because the model over-thinks past the
cap. "Robust to quantization" becomes "tool calls are robust; structured output is
truncation-limited" — a sharper, more honest claim.

Fix:

```python
ANSWER = re.compile(r"</think>", re.I)
def final_answer(out):
    parts = ANSWER.split(out, maxsplit=1)
    return parts[1] if len(parts) > 1 else None   # None => no answer emitted
```

Grade only the tail, and hard-fail with `detail="truncated_no_answer"` when it's
`None`. Also record `stop_reason` (eos vs length) per case — that distinction is
currently invisible in the artifacts.

### B2. `jsonschema` is dev-only but used at runtime, and the fallback passes

`grading.py`:

```python
except ImportError:
    return {"pass": True, "detail": "json_object (schema unchecked)"}
```

`jsonschema` sits in `[project.optional-dependencies].dev`. So a plain
`pip install nanbeige-mlx-eval` silently passes **every** `json_schema` case. Move it
to `dependencies`, and make the fallback raise rather than pass — a grader that can't
grade must never return `pass`.

### B3. `mean_tps` isn't a throughput

Per-case tok/s averaged across 58–384-token generations. Short generations are
dominated by fixed per-call overhead so they look slow; the truncated 384-token case
looks fast (34.5 tok/s) and drags the mean up. The 4-bit headline of **21.4** comes
from that mean; on comparable short cases the steady-state number is **~16–17**.

Report `total_generated_tokens / total_generation_time`, or a median. And split the
TTFT column: tool cases run ~2.3 s, the bare-prompt case ~0.55 s — averaging two
prompt regimes into one number ("2.09 s") describes neither.

### B4. `peak_rss_mb` is a process-wide high-water mark

`resource.getrusage(RUSAGE_SELF).ru_maxrss` is monotonic, so it's identical for all 8
cases in every run (3167.1 across the board) — per-case values carry no information.
The README then *averages* peaks across the EN and ZH processes (3700, 4023), which
isn't the peak of anything.

Use `mx.get_peak_memory()` with `mx.reset_peak_memory()` between cases, and report the
max. `ru_maxrss` also doesn't reliably capture Metal buffer allocations, which is most
of what you want to measure.

### B5. `mlx_version: "unknown"` in every single manifest

```python
import mlx
mlx_version = getattr(mlx, "__version__", "unknown")   # mlx has no __version__
```

It's `mlx.core.__version__`. Your headline reproducibility field has never once
recorded a value. Also absent from the manifest: mlx-lm version, quant bits and group
size, git commit, suite content hash, weights hash. `model` is a bare relative path
(`models/nanbeige-mlx-4bit`) — nothing ties a run to the bytes it ran against.

### B6. n=8, single run, no warmup

8/8 on n=8 is a Wilson 95% CI of roughly **[67%, 100%]**. `pass_rate` is rounded to
four decimals. One run per config, no warmup pass (first case eats Metal kernel
compilation plus the `mx.compile` on `swiglu`), no repeats, no median. The prose
hedges ("directional, not benchmark-grade") but the tables present bolded **8 / 8**
and three-significant-figure tok/s. Either add CIs and a median-of-3, or drop the
decimals so the format matches the claim.

### B7. "Answer-key protection" isn't a thing here

`_write_results` avoids persisting expected answers — but `suites/*.json`, containing
every expected answer, is committed to the repo. Drop the claim; it reads as security
theatre and invites a reader to check.

### B8. "Greedy" is asserted but never set

`stream_generate` is called with no `sampler`, relying on mlx-lm's default while
`manifest.settings.decoding` hardcodes `"greedy"`. Pass
`sampler=make_sampler(temp=0.0)` explicitly so the manifest is a fact rather than a
hope. Also: `prompt_tokens` is computed from your own `tokenizer.encode(...)` while
`stream_generate` re-encodes the string itself — those can disagree, and your tps
math uses yours.

### B9. `enable_thinking` is never controlled

The template appends `<think>\n` unconditionally, so every case pays the CoT tax and
the 384-cap case is a coin flip. Make it an explicit per-suite field and report both
modes — "how much does thinking cost, and does disabling it hurt tool selection?" is a
better result than either mode alone.

---

## Packaging and distribution

**C1. `mlx-lm>=0.20` is far too loose.** The port imports
`mlx_lm.models.activations.swiglu`, `mlx_lm.models.rope_utils.initialize_rope`, and
depends on the `model_file` config hook — all much newer than 0.20. Anyone resolving
0.2x gets an `ImportError`. Pin `>=0.31` (what you tested).

**C2. The shipped `model_file` depends on four mlx-lm internal modules.** The docstring
says "public model helpers", but `mlx_lm.models.*` carries no stability guarantee.
This file is baked into every weight repo you publish — an upstream refactor
permanently breaks quants users already downloaded, and you cannot patch their copies.
Inline the ~20 lines (`swiglu`, causal mask, the sdpa call) so the published
`model_file` has zero internal deps. This is the highest-leverage durability fix in
the whole project.

**C3. `prepare_source` mutates the user's source directory.** It rewrites their
`config.json` and drops `nanbeige.py` into the HF download. If `--src` points at an
HF cache snapshot, you've corrupted it (hash mismatch on next verify). Copy to a temp
dir, or symlink weights and write config to the copy.

**C4. `nanbeige-mlx-eval convert` never calls `prepare_source`.** `cmd_convert` goes
straight to `to_mlx`, which requires `model_file` already in the config — so the exact
command in the README fails on a fresh download. (`python -m mlx_nanbeige.convert`
does it right; the eval CLI doesn't.)

**C5. Converted repos lose tokenizer files.** Output `tokenizer_config.json` is 492
bytes vs 10,976 in the source. Gone: `added_tokens.json`, `special_tokens_map.json`,
`tokenizer.model`. Leaked in: transient keys `is_local`, `local_files_only`,
`backend`. Fast-tokenizer loads still work (added tokens live inside `tokenizer.json`),
but `use_fast=False` breaks with `tokenizer_class: LlamaTokenizer` and no
`tokenizer.model`. The chat template survives only as `chat_template.jinja`, which
needs a recent transformers to be read. `model_max_length` is the int64 sentinel, not
262144. Copy the tokenizer files through verbatim and add a post-convert assertion
that `AutoTokenizer.from_pretrained(out)` round-trips `<|im_end|>` → 166101.

**C6. `--max-kv-size` silently no-ops.** `make_prompt_cache` ignores `max_kv_size`
whenever `make_cache` exists, so the hook that makes the looped architecture work also
disables rotating-cache support. `--kv-bits` still works (`KVCache.to_quantized`).
Worth a README note, because the arithmetic is stark: 44 slots × 8 KV heads × 128 dim
× 2 (K+V) × 262,144 positions × 2 bytes ≈ **47 GB** of KV at full context. The 262K
number in your spec table is unreachable on a 16 GB machine, and the one knob that
would help is the one you disabled. Say so — it's a real and interesting cost of the
looped design that nobody else has written down.

**C7. The uploaded model card has no license.** mlx-lm's generated `README.md` is:

```yaml
language: en
library_name: mlx
pipeline_tag: text-generation
tags: [mlx]
```

No `license:`, no `base_model:`, and `language: en` for a bilingual model. An
unlicensed weights repo is both an Apache-2.0 §4 compliance gap and something HF
flags. `upload.py` should write proper frontmatter (`license: apache-2.0`,
`base_model: Nanbeige/Nanbeige4.2-3B`, `language: [en, zh]`), copy the upstream
LICENSE, and add a NOTICE stating the modification (quantized to N-bit for MLX).

**C8. `.zcode/` is untracked and not in `.gitignore`.** Add it.

**C9. Internal references leak into public output.** `cli.py` and `suite.py` both
mention *"the sibling `ornith-mlx-eval` project"*, and `cli.py`'s docstring is the
`argparse` description — so it prints on `--help`.

**C10. `model.py` exists in five places, byte-identical but for one docstring line**
(`mlx_nanbeige/model.py`, `nanbeige_mlx_eval/models/nanbeige.py`, and one copy inside
each of the three quant dirs). Pick one home; have the eval project depend on
`mlx-nanbeige` and import from it. Divergence here is a silent-wrong-results bug.

**C11. `benchmark_results/` is gitignored, but the README publishes numbers from it.**
Commit the artifacts you cite, or the reader can't check them. (I could only check
them because you have them locally.)

---

## Your question: bundle the weights?

**Legally, yes — with conditions, and not as MIT.**

- Nanbeige4.2-3B is **Apache-2.0** ([model card](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)),
  which permits redistribution of derivative works. Quantization is a modification, so
  §4 applies: include the upstream LICENSE, retain notices, and state that you changed
  the files.
- **You cannot relicense the weights MIT.** MIT covers your code. A bundle labelled
  "MIT" would be a false license claim. Keep them separate: MIT for `mlx_nanbeige/`,
  Apache-2.0 on the weights repo. Your current READMEs already draw this line
  correctly — don't blur it by merging the artifacts.

**Practically, no — don't put them in the pip package.**

PyPI's default limits are **100 MB per file** and 10 GB per project
([storage limits](https://docs.pypi.org/project-management/storage-limits/)). Your
4-bit checkpoint is 2.2 GB — 22× the per-file cap. Increases are granted case by case
and nobody is approving a 2.2 GB wheel for a model port. Even with an exemption,
you'd lose resumable downloads, delta updates, revision pinning, and per-quant choice,
and `pip` would re-download the whole thing on every version bump.

**Do this instead — it gets you the UX you want:**

Publish each quant as its own HF repo. You already wrote `upload.py`:

```bash
python -m mlx_nanbeige.upload --model-dir models/nanbeige-mlx-4bit \
  --repo-id jishnuvenugopal/Nanbeige4.2-3B-mlx-4bit
```

Then the user experience is already a one-liner, with no conversion step:

```python
import mlx_lm
model, tok = mlx_lm.load("jishnuvenugopal/Nanbeige4.2-3B-mlx-4bit")
```

Weights arrive on first use, cached in `~/.cache/huggingface`, resumable,
revision-pinnable. From the user's side that's indistinguishable from bundling — and
it's exactly how `mlx-community` ships every model. If you want it to feel more
first-class, add a thin `mlx_nanbeige.pull("4bit")` that wraps `snapshot_download`
with the repo ids baked in, and put `--kv-bits` guidance in the model card.

Fix C7 (license frontmatter + NOTICE) before you push any weights repo.

---

## Implementation plan

### Decision 0: settle the repo boundary first

Almost everything below depends on this, so do it before writing any code. Right now
`model.py` lives in five places and both repos own conversion. Split it:

| | `mlx-nanbeige` (the port) | `nanbeige-mlx-eval` (the study) |
|---|---|---|
| owns | `model.py`, `convert.py`, `upload.py`, `pull.py` | suites, grading, runtime, reporting, parity, CLI |
| publishes to | PyPI + HF weight repos | GitHub only |
| depends on | mlx, mlx-lm, huggingface_hub | **mlx-nanbeige**, transformers, jsonschema |
| audience | anyone running Nanbeige on a Mac | readers of your writeup |

Concretely:

1. `mlx-nanbeige` keeps `mlx_nanbeige/model.py` as the single source of truth.
2. Delete `nanbeige_mlx_eval/models/nanbeige.py` and `nanbeige_mlx_eval/convert.py`.
3. Add `mlx-nanbeige>=0.2.0` to the eval's `dependencies`; replace
   `from .models.nanbeige import Model, ModelArgs` with
   `from mlx_nanbeige.model import Model, ModelArgs`, and `cmd_convert` with a
   passthrough to `mlx_nanbeige.convert.to_mlx` (which will call `prepare_source`
   itself after P3.3).
4. Keep the three copies inside `models/nanbeige-mlx-*bit/` — those are *supposed* to
   be there, they're the shipped `model_file`. They're generated, not source.

This resolves **C10** and makes **C4** disappear.

Rough total for everything below: **3–4 focused days.** The first day buys most of the
credibility.

---

### Phase 1 — Eval integrity (½ day, `nanbeige-mlx-eval`)

Do this first. It's cheap, it's the fix that changes published numbers, and doing it
before the fidelity work means you never have to re-run the ladder twice.

**P1.1 — Grade only the final answer** (fixes **B1**)

New in `grading.py`:

```python
_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)

def split_reasoning(output: str) -> tuple[str, str | None]:
    """Return (reasoning, answer). answer is None if the block never closed."""
    parts = _THINK_CLOSE.split(output, maxsplit=1)
    if len(parts) == 1:
        return ("", None) if "<think" not in output.lower() else (output, None)
    return parts[0], parts[1]
```

Then in `grade()`, before dispatching:

```python
def grade(expect, output, *, require_answer=True):
    reasoning, answer = split_reasoning(output)
    if require_answer and answer is None:
        return {"pass": False, "detail": "truncated_no_answer",
                "reasoning_chars": len(reasoning)}
    target = answer if answer is not None else output
    ...  # dispatch on target, never on `output`
```

Every grader now sees `target`, not the raw stream. Add `require_answer` as a
per-suite override so a non-thinking suite still works.

**P1.2 — Record why generation stopped** (fixes the invisible-truncation problem)

In `MLXRuntime.run_case`, mlx-lm's `GenerationResponse` carries `finish_reason`.
Capture the last one and put `stop_reason: "stop" | "length"` into both `CaseResult.meta`
and the `results.jsonl` row. Then a truncation is visible in the artifact without
grepping the output. Add to `summary.json`:

```json
"truncated": {"n": 1, "ids": ["zh-json-profile"]}
```

**P1.3 — A grader that can't grade must not pass** (fixes **B2**)

Move `jsonschema` from `[dev]` into `dependencies`. Change the fallback:

```python
except ImportError as exc:
    raise RuntimeError(
        "json_schema grading requires `jsonschema`; install it or remove "
        "json_schema cases from the suite."
    ) from exc
```

**P1.4 — Regenerate the tables**

`report`/`compare` already read only persisted artifacts, so:

```bash
for d in benchmark_results/*-mlx-*; do nanbeige-mlx-eval report "$d"; done
```

…won't help, because grading happens at run time, not report time. Two options:

- *Preferred:* add `nanbeige-mlx-eval regrade <run_dir>` that re-grades the persisted
  `output` field against the suite and rewrites `results.jsonl` + `summary.json` +
  `report.md`. This makes grading changes retroactive forever, which is worth having
  as permanent infrastructure — you'll change the grader again.
- Or just re-run the ladder (~6 runs × ~1 min). Fine now; won't scale.

Either way the published table becomes 4-bit 8/8 + 7/8, 6-bit 7/8 + 7/8, 8-bit
7/8 + 7/8, and the README narrative shifts to "tool calls are quantization-robust;
structured output is truncation-limited." Also raise `max_tokens` on the two
`json-profile` cases to 1024 and re-run, so you separate "can't do it" from
"didn't have room" — report both.

**P1.5 — Tests**

- `test_grade_rejects_unclosed_think` — output with `<think>` and no `</think>`
  containing valid JSON must **fail** with `truncated_no_answer`.
- `test_grade_ignores_reasoning_block` — correct tool call in the reasoning, wrong one
  after `</think>` → must fail. And the mirror: garbage in reasoning, correct call
  after → must pass.
- Replay the actual 8-bit EN output as a fixture; assert it fails. That's a
  regression test against the exact bug.

---

### Phase 2 — Close the fidelity question (1 day, `nanbeige-mlx-eval`)

**P2.1 — Make the reference device a flag, default `cpu`** (fixes **A1**)

In `parity.py::_gather_hf_logits`, add `device: str = "cpu"` and `dtype` params; thread
`--device {cpu,mps}` and `--dtype {bf16,fp32}` through `cmd_parity`. Record both in the
output JSON. bf16 on CPU is ~8.3 GB — fits your 16 GB, and you already free the HF
model before loading MLX.

Expected outcome: mean cosine goes from 0.844 to >0.99. If it does, the README section
rewrites itself and you delete the "couldn't complete fp32 isolation" caveat entirely.

**P2.2 — Isolate RoPE precision as its own measurement**

The reference downcasts cos/sin to bf16; `mx.fast.rope` stays fp32. Quantify it rather
than arguing about it: take one layer's `q_proj` output, apply (a) `mx.fast.rope`,
(b) the reference path in fp32, (c) the reference path with cos/sin cast to bf16.
Report all three pairwise. If (a)↔(c) is your 0.016 and (a)↔(b) is ~1e-6, you've proved
the port is *more* accurate than the reference and the gap has a known floor. That's a
much better sentence than the one currently in the README.

**P2.3 — Per-layer divergence trace** (fixes **A2**, half of it)

New `nanbeige_mlx_eval/trace.py` + `nanbeige-mlx-eval trace --src ... --layer-by-layer`.

**Gotcha worth knowing before you build it:** `output_hidden_states=True` will *not*
give you 44 states. In `NanbeigeModel.forward`, `last_loop_all_hidden_states =
current_loop_all_hidden_states` overwrites on every loop iteration, so you only ever
get the final loop's 22. To capture all 44, register forward hooks on
`model.model.layers[i]` — each fires twice per forward, so append in call order:

```python
captured = []
hooks = [l.register_forward_hook(
            lambda m, i, o: captured.append(o[0].detach().float().cpu()))
         for l in hf_model.model.layers]
hf_model(input_ids, use_cache=False)
# captured is now 44 long: [loop0_l0..loop0_l21, loop1_l0..loop1_l21]
```

On the MLX side, walk `model.model.layers` manually in the same two-loop order,
applying `self.norm` at each boundary, and collect. Emit cosine + max-abs per effective
layer to `trace.json` and a small markdown table.

Acceptance: a monotone, gently-decaying curve = numerics. A step change at layer *k* =
a bug in layer *k*. Either way you stop guessing. Note that this command is also how
you'd debug the *next* architecture you port, so it earns its keep.

**P2.4 — Prefill-vs-decode self-consistency** (fixes **A2**, the important half)

This one needs no torch, runs in seconds, and is the test the 44-slot cache actually
demands. Put it in **`mlx-nanbeige`** as `tests/test_cache_consistency.py`, since it
tests the port not the study:

```python
def test_prefill_matches_incremental_decode(tiny_args):
    """Two virtual cache slots per layer must stay in lockstep."""
    model = Model(tiny_args)          # small random model: 2 layers, num_loops=2
    mx.eval(model.parameters())
    ids = mx.array([[5, 9, 3, 7, 2, 8]])

    full = model(ids)[0, -1]          # one-shot prefill, no cache

    cache = model.make_cache()        # incremental, 44-slot path
    for i in range(ids.shape[1] - 1):
        model(ids[:, i:i+1], cache=cache)
    step = model(ids[:, -1:], cache=cache)[0, -1]

    assert mx.allclose(full, step, atol=2e-2).item()
```

Also assert `all(c.offset == ids.shape[1] for c in cache)` — every one of the 44 slots
must advance in lockstep. Swap the loop index in `NanbeigeModel.__call__` and confirm
the test goes red; if it doesn't, the test is wrong.

Add a second one: `test_cache_offset_matches_across_loops` — after a forward, slots
`0..21` and `22..43` must have identical offsets.

---

### Phase 3 — Harden the port before you publish weights (½ day, `mlx-nanbeige`)

Sequencing matters here: `model.py` gets **frozen into every weight repo you push**.
Fix it before uploading, not after.

**P3.1 — `o_proj` bias** (fixes **A3**)

```python
self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)
```

**P3.2 — Reduce the shipped file's dependency surface** (fixes **C1**, **C2**)

Don't blanket-inline — you'd lose quantized-KV dispatch. Split by whether the helper
carries real behavior:

*Inline (trivial, zero downside):*

```python
def _swiglu(gate, x):
    return nn.silu(gate) * x

def _causal_mask(h, cache):
    if cache is not None and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
    return None if h.shape[1] == 1 else "causal"
```

Also inline `BaseModelArgs.from_dict` (it's a dataclass-field filter, ~6 lines) and the
`rope_scaling is None` branch of `initialize_rope` (`nn.RoPE(dims, traditional, base)`).
Keep `initialize_rope` behind a guarded import for the scaled variants.

*Guard, don't inline* — `scaled_dot_product_attention` and `KVCache` carry
`QuantizedKVCache` dispatch you want to keep:

```python
try:
    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.cache import KVCache
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mlx_nanbeige requires mlx-lm >= 0.31 (tested 0.31.3); the internal "
        "helpers this model_file uses moved or were removed in your version."
    ) from exc
```

An actionable error beats a raw `ModuleNotFoundError` in a stranger's traceback three
years from now. Then bump `pyproject.toml` to `mlx-lm>=0.31` and `mlx>=0.32`, and
delete the "depends only on public helpers" line from the docstring — say
"pinned to mlx-lm ≥ 0.31 internals; see P3.2" instead. Honesty in the file that ships.

**P3.3 — Stop mutating the source directory** (fixes **C3**)

```python
def prepare_source(src_dir, model_file=MODEL_FILE, workdir=None):
    """Stage a conversion-ready copy. Never writes into src_dir."""
    src = Path(src_dir)
    stage = Path(workdir or tempfile.mkdtemp(prefix="nanbeige-stage-"))
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, stage / f.name)
    for f in src.glob("*.safetensors"):
        (stage / f.name).symlink_to(f.resolve())   # don't duplicate 8 GB
    shutil.copy(model_file, stage / "nanbeige.py")
    cfg = json.loads((stage / "config.json").read_text())
    cfg["model_file"] = "nanbeige.py"
    (stage / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    return stage
```

Symlinks keep it free. `to_mlx` calls `prepare_source` itself and cleans up the stage
dir, so `convert` is one step and idempotent from a pristine HF snapshot.

**P3.4 — Preserve tokenizer files + assert the round-trip** (fixes **C5**)

After `convert()`, copy through anything mlx-lm dropped and verify:

```python
CARRY = ["added_tokens.json", "special_tokens_map.json", "tokenizer.model",
         "chat_template.jinja"]

def _finalize(src: Path, out: Path) -> None:
    for name in CARRY:
        if (src / name).exists() and not (out / name).exists():
            shutil.copy2(src / name, out / name)
    # drop transient keys mlx-lm leaked from tokenizer.init_kwargs
    tc = out / "tokenizer_config.json"
    cfg = json.loads(tc.read_text())
    for k in ("is_local", "local_files_only", "backend"):
        cfg.pop(k, None)
    cfg["model_max_length"] = 262144
    tc.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

def verify(out: Path) -> None:
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(str(out))
    assert t.convert_tokens_to_ids("<|im_end|>") == 166101, "eos token lost"
    assert t.eos_token_id == 166101
    rendered = t.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
    assert "<|im_start|>assistant" in rendered and "<think>" in rendered
```

Call `verify` at the end of `to_mlx` and print a one-line PASS. A conversion that
silently loses the EOS token produces a model that never stops — worth 10 lines to
never debug that.

**P3.5 — Document the KV-cache ceiling** (fixes **C6**)

Two sentences in both READMEs, plus a `NotImplementedError`-free but explicit note in
`make_cache`'s docstring: `--max-kv-size` is inert because `make_prompt_cache` skips it
whenever `make_cache` exists; `--kv-bits` works. Include the arithmetic — 44 slots ×
8 KV heads × 128 × 2 × 262,144 × 2 B ≈ **47 GB** at full context. This is a genuine,
quantifiable cost of the looped design and nobody else has written it down. It belongs
in your writeup as a finding, not buried as a caveat.

---

### Phase 4 — Honest performance numbers (½ day, `nanbeige-mlx-eval`)

**P4.1 — Real throughput, not a mean of ratios** (fixes **B3**)

In `_write_summary`, replace `mean_tps`:

```python
gen_tokens = sum(r["generated_tokens"] for r in mlx_rows)
gen_time   = sum(r["total_s"] - r["ttft_s"] for r in mlx_rows)
summary["latency"] = {
    "decode_tps_aggregate": round(gen_tokens / gen_time, 2),   # headline
    "decode_tps_median":    round(statistics.median(tps_list), 2),
    "ttft_s_median":        round(statistics.median(ttft_list), 3),
    "ttft_s_by_regime":     {"with_tools": ..., "bare_prompt": ...},
}
```

Note `total_s - ttft_s` is the actual decode window; dividing by `total_s` (what you do
now) folds prefill into decode speed. Drop `mean_tps` from the schema entirely so it
can't be cited again by accident.

**P4.2 — Real memory** (fixes **B4**)

```python
import mlx.core as mx
mx.reset_peak_memory()          # before each case
...
peak_bytes = mx.get_peak_memory()
```

Report per-case and take the **max** across cases, never the mean. Keep `ru_maxrss` as
a secondary field if you like, labelled `process_rss_mb`, but the MLX number is the one
that means something.

**P4.3 — Warmup + repeats** (fixes **B6**)

Add `--warmup 1` (default 1) that runs case 0 and discards it, and `--repeats 3`
(default 1) taking the median per case. Record both in the manifest. First-case Metal
kernel compilation plus `mx.compile` on `swiglu` currently lands in your published
numbers.

**P4.4 — Wilson interval on every pass rate** (fixes **B6**)

```python
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z*z/n
    c, m = p + z*z/(2*n), z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c - m)/d, (c + m)/d)
```

Put `pass_rate_ci95` in `summary.json` and render it in `report.md` and the README
table: **8/8 → 1.00 [0.67, 1.00]**. That single bracket does more for the writeup's
credibility than any amount of hedging prose, and it makes the case for P4.5 obvious.

**P4.5 — Widen the suite**

Your own next-step, and the CI above shows why. Target ~30 cases per language. Highest
value additions, roughly in order:

1. **Negative cases** — a request no provided tool can serve; the model must answer in
   natural language rather than force a call. Currently untested, and it's the most
   common real-world agentic failure.
2. **Missing-argument cases** — the template's system prompt explicitly instructs the
   model to *ask* for missing parameters. Test that.
3. **Multi-tool disambiguation** — 4+ similar tools, one correct.
4. **Multi-turn with `<tool_response>`** — the template has a whole branch for tool
   results that nothing currently exercises.
5. **Thinking on/off pairs** — same case both ways (needs **B9** first).

At n≈30, 30/30 gives a CI of [0.88, 1.00] — a claim worth making.

**P4.6 — Explicit sampler and thinking control** (fixes **B8**, **B9**)

```python
from mlx_lm.sample_utils import make_sampler
self._sampler = make_sampler(temp=0.0)      # assert, don't assume
```

Pass it to `stream_generate`. Add `enable_thinking: bool | None` as a suite/case field,
thread it into `apply_chat_template`, and record it in the manifest.

**P4.7 — Fix the manifest** (fixes **B5**)

```python
def _env_info():
    import mlx.core as mx
    from importlib.metadata import version, PackageNotFoundError
    def v(pkg):
        try: return version(pkg)
        except PackageNotFoundError: return "absent"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": mx.__version__,                    # not mlx.__version__
        "mlx_lm": v("mlx-lm"),
        "mlx_nanbeige": v("mlx-nanbeige"),
        "transformers": v("transformers"),
        "git_commit": _git_sha(),
    }
```

Plus, in the manifest: `quantization` (bits + group size, read from the model's
`config.json`), `suite_sha256`, and `weights_sha256` of `model.safetensors`. A run
should be traceable to exact bytes. Add `test_manifest_has_no_unknown_versions` so this
can't silently rot again.

---

### Phase 5 — Publish (½ day, `mlx-nanbeige`)

Gate: Phases 2 and 3 green. Do not push weights containing a `model.py` you're still
editing.

**P5.1 — Proper model card** (fixes **C7**)

`upload.py` writes `README.md` before uploading:

```yaml
---
license: apache-2.0
base_model: Nanbeige/Nanbeige4.2-3B
language: [en, zh]
library_name: mlx
pipeline_tag: text-generation
tags: [mlx, nanbeige, looped-transformer, apple-silicon]
---
```

Body: what changed (converted to MLX, quantized to N-bit, group size 64), the one-liner
load snippet, the measured numbers with CIs, a link back to the eval repo, and an
Apache-2.0 §4 notice. Also `shutil.copy` the upstream LICENSE to `LICENSE` and write a
`NOTICE` stating the modification. Add a `--dry-run` that renders the card and lists
files without uploading.

**P5.2 — `pull` helper** (makes it feel bundled, per your question)

```python
QUANTS = {
    "4bit": "jishnuvenugopal/Nanbeige4.2-3B-mlx-4bit",
    "6bit": "jishnuvenugopal/Nanbeige4.2-3B-mlx-6bit",
    "8bit": "jishnuvenugopal/Nanbeige4.2-3B-mlx-8bit",
}

def pull(quant: str = "4bit", revision: str | None = None) -> str:
    """Download a published quant; returns a local path for mlx_lm.load."""
    return snapshot_download(QUANTS[quant], revision=revision)
```

So the README's headline becomes two lines with no conversion step:

```python
from mlx_nanbeige import pull; import mlx_lm
model, tok = mlx_lm.load(pull("4bit"))
```

**P5.3 — Publish order:** 4-bit first, load it from a clean venv on a different
machine (or at least a fresh `HF_HOME`) with only `pip install mlx-nanbeige`, confirm
it generates, then push 6-bit and 8-bit. Tag `v0.2.0` and pin the revision SHAs in the
eval's README so the study is reproducible against exact weights.

---

### Phase 6 — Hygiene (1–2 hours)

- **C8** — add `.zcode/` to both `.gitignore`s.
- **C9** — drop the two `ornith-mlx-eval` references; `cli.py`'s docstring is `--help`
  text. Give `argparse` its own short description instead of `__doc__`.
- **C11** — un-ignore `benchmark_results/`, or add a committed
  `results/published/` holding exactly the runs the README cites (they're small —
  `results.jsonl` is a few KB). Numbers a reader can't open are numbers they can't trust.
- Replace `plan.md`-era wording in the eval README's "Local-only files" section once
  `benchmark_results/` is partly public.
- The docstring in `parity.py` claims *"the bar is high agreement (logit cosine ≈ 1)"*
  while the code reports 0.844 as a result. After P2.1, make that a real gate: exit
  non-zero if `mean_cosine < 0.99`, so "fidelity gate" is a gate.

---

### What the README says when this is done

Worth writing the target claims now, so you know when you're finished:

- **Fidelity:** "Mean logit cosine 0.99x vs the HF reference (bf16, CPU). Divergence
  traced layer-by-layer across all 44 effective layers; the residual is accounted for
  by the reference's bf16 downcast of RoPE cos/sin, measured in isolation at N. The
  44-slot loop-aware cache is verified by prefill-vs-incremental-decode equality."
  — a specific, falsifiable claim instead of an apology.
- **Capability:** "Tool selection and argument extraction hold at 29/30 EN
  [0.83, 1.00] and 28/30 ZH from 4-bit up. Structured output is truncation-limited,
  not capability-limited: at a 384-token cap the model runs past it mid-reasoning in
  4/6 configurations; at 1024 it succeeds in 6/6."
- **Cost:** "~150 CoT tokens and ~2.3 s TTFT per tool call. Full 262K context is
  unreachable on 16 GB — the looped design needs 44 KV slots, ~47 GB at max context —
  and `--max-kv-size` is inert when a model supplies `make_cache`."

That last one is a real contribution. The 47 GB figure and the `make_cache`/
`max_kv_size` interaction are things you found by porting a weight-shared architecture,
and they'd be news to most people writing mlx-lm ports. Lead with it.

---

Sources: [Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) ·
[PyPI storage limits](https://docs.pypi.org/project-management/storage-limits/)

---
---

# Addendum — post-implementation review (2026-07-26)

Everything in Phases 1–6 landed and the eval work is solid: the grader fix is
correct and regression-tested, the metrics are honest, the manifest is real, the
package split is clean, and the ladder re-run at 1024 tokens is a genuinely better
result than what was published before. Two things need revisiting.

## 1. I was wrong about A1, and the replacement conclusion is also wrong

My hypothesis was that MPS bf16 caused the gap and CPU would clear 0.99. **You
falsified it properly** — CPU gives 0.847 against MPS's 0.844. Device numerics are
ruled out. That's a real result and the right way to get it.

But the README now concludes:

> the divergence is present at layer 0 and has no step discontinuity (no bug
> signature); it is numerics

**The trace data does not support that.** Two independent reasons.

### The curve is not monotone, and it *does* have a step

| effective layer | 0 | 3 | 20 | **21** | **22** | 43 |
|---|---|---|---|---|---|---|
| cosine | 0.925 | 0.834 | 0.956 | **0.997** | **0.895** | 0.980 |

Agreement *falls* over layers 0–3, then *rises* over layers 3–21, then **drops
0.10 in a single step at the loop boundary**, then rises again. Compounding
numerical error is monotone in agreement — it cannot improve over 18 of 22 layers.
And layer 21 → 22 is a step discontinuity, sitting at exactly the one place in the
forward pass where something structural happens: `self.norm` at the loop boundary.
The heuristic in my own plan ("monotone decay = numerics; a step = a bug") was
applied to a curve that is neither monotone nor step-free.

### The magnitude is ~100× too large for bf16

Layer 0 receives an **identical** input on both sides (the same embedding lookup of
the same token ids). Its output agrees at cosine 0.925. For unit vectors,
cos θ = 0.925 → θ ≈ 22°, so the error vector is roughly `2·sin(11°) ≈ 38%` of the
output norm — after *one* layer, from *identical* inputs. bf16 carries ~8 mantissa
bits; per-op relative error is ~0.4%, and matmul accumulation is fp32 on both
Metal and PyTorch CPU. 38% is not bf16.

### What the curve actually says

Cosine of a hidden state tracks `|residual| / |error|`. Within a loop the residual
stream grows monotonically (every block adds to it), so a **constant absolute
per-layer error** shows up as steadily *improving* cosine — which is exactly what
layers 3→21 do. At the loop boundary `self.norm` rescales the residual back down to
~√d, the error becomes a large fraction again, and agreement crashes. That is the
21→22 step, and it also explains why layer 0 (residual = bare embedding, smallest it
ever is) and layer 22 (residual just renormalised) are the two worst points in the
whole trace.

So the signature isn't "compounding numerics." It's **a roughly constant,
systematic per-layer discrepancy**, diluted by residual growth and re-exposed by
each norm. That is a much narrower thing to hunt, and it's hunted in one layer.

## 2. The decisive experiment was never run, and its stated blocker is false

Both the old and new READMEs cite the same obstacle:

> full fp32 isolation that would distinguish "logic" from "numerics" bit-exactly
> could not complete: a 16 GB machine cannot hold the 16 GB fp32 reference

True for the whole model, and irrelevant to the question. The question is about a
*single layer*, and one Nanbeige layer is 143 M params:

```
q 3072x6144 + k,v 3072x1024 x2 + o 6144x3072
  + gate,up 3072x10752 x2 + down 10752x3072   =  143.1 M
```

**573 MB in fp32. ~1.2 GB for both frameworks together.** The experiment declared
impossible costs about a gigabyte and two minutes.

I've added `nanbeige_mlx_eval/bisect.py` + a `bisect` subcommand. It feeds the same
input to both frameworks and compares 14 stages **independently from a common
input** (not chained, so an early gap can't mask or manufacture a later one):
`input_layernorm → q/k/v_proj → rope_q/rope_k → attn_out → o_proj →
post_attention_layernorm → gate/up/swiglu/down → block`.

```bash
# 1. Is it logic?  Any fp32 stage below ~0.99999 is a bug, full stop.
nanbeige-mlx-eval bisect --src models/nanbeige42-hf --dtype fp32 --gate \
  --out results/published/bisect_fp32.json

# 2. If fp32 is clean — which operation loses it at bf16?
nanbeige-mlx-eval bisect --src models/nanbeige42-hf --dtype bf16

# 3. Test the specific hypothesis below.
nanbeige-mlx-eval bisect --src models/nanbeige42-hf --dtype bf16 --bf16-rope
```

Run #1 before writing another sentence about this in the README. It is the only
measurement that separates the two explanations, and it now costs nothing.

## 3. The leading hypothesis, and why the RoPE result was misread

`rope_precision` reports:

```json
{"reference_fp32_cos_vs_bf16_cos": {"cosine": 0.999997, "max_abs": 0.02242}}
```

and the README treats it as a negligible floor. **The cosine there is the wrong
statistic.** cos/sin vectors are dominated by near-1 entries, so cosine stays at
0.999997 no matter how badly you round them; `max_abs = 0.022` is the number that
matters, and it's a ~2% elementwise perturbation.

That 2% lands on **both q and k**, then goes through a 128-term dot product. Rounding
is systematic, not random, so the errors don't cancel — they accumulate coherently
into the attention logits, and softmax converts a ~1-unit logit shift into materially
different attention weights. Every layer, identically. Device-independent, which is
precisely why moving to CPU changed nothing.

This hypothesis predicts everything observed: constant per-layer error, worst where
the residual is smallest, unaffected by MPS-vs-CPU, and behaviourally harmless
(attention still attends to the right tokens; only the logit tail moves).

`--bf16-rope` tests it in one shot by applying the reference's downcast to **both**
sides. If the bf16 block cosine jumps, you're done — and the conclusion is a strong
one: *the port is more accurate than the reference, and the gap is the reference's
own precision loss, quantified.* That's a far better README line than "we couldn't
close it."

If it doesn't jump, run #1 and the bisect will name the stage.

## 4. Live bug in the new code

`trace._gather_mlx_layer_states` accepted `dtype` and never used it — the MLX model
was always loaded at the safetensors' bf16. So `trace --dtype fp32` was comparing an
**fp32 reference against a bf16 port** and reporting the difference as divergence.
Anyone running the documented fp32 mode would have gotten a spurious result and
concluded the port was broken. Fixed (`model.set_dtype(...)`).

Also fixed: the parity `interpretation` string still asserted "On CPU/bf16 the mean
cosine should clear 0.99" — a prediction the run disproved, now baked into the
published artifact. It now states what actually happened.

## 5. Smaller items

- **The parity probes are weak.** Four of six prompts have `top1_hf == 13` (a
  newline). They're raw completions with no chat template, so you're measuring the
  logit tail of a low-entropy "the prompt just ended" distribution — the regime where
  relative error is largest and least meaningful. Add chat-templated prompts with
  sharp, content-bearing next tokens (a factual completion, a forced JSON opening
  brace, a mid-sentence continuation). `mean_kl = 2.06 nats` is a big number; it
  deserves probes that make it interpretable.
- **P4.5 was not done.** Suites are still 8 cases. The CI you now correctly report,
  `[0.68, 1.00]`, is the argument for widening — 8/8 at n=8 is compatible with a true
  rate of 68%. The four case types I'd add first are in the plan above (negative
  cases, missing-argument, multi-tool disambiguation, multi-turn `<tool_response>`).
  Negative cases matter most: nothing currently tests that the model *declines* to
  call a tool it shouldn't.
- **`repeats: 1` in every published manifest.** The plan called for `--repeats 3`;
  the flag works, it just wasn't used. Perf numbers are still single-shot.
- **`git_commit: "0a0cf2c"` is the pre-change commit** in all six manifests, because
  nothing has been committed yet. The published artifacts currently point at a commit
  that doesn't contain the code that produced them. Commit first, then re-stamp (or
  re-run) so the provenance is real.
- **`decode_tps_aggregate` 31.6 vs `decode_tps_median` 17.1** is a 1.8× spread. Both
  are correct and they measure different things — aggregate is throughput on a mixed
  workload, median is what a single short call feels like. The README should say which
  one it's quoting and why, or a reader will assume the flattering one was chosen.

## 6. Corrected README wording

Until run #1 comes back, the fidelity section should say what's known and no more:

> **Fidelity: unresolved, and honestly so.** Mean logit cosine is 0.847 against the
> HF reference (top-1 83%). The MPS hypothesis is falsified — CPU reproduces the same
> number. The 44-layer trace shows the divergence is present at layer 0 from identical
> inputs, is roughly constant per layer, and steps by 0.10 at the loop boundary where
> `norm` rescales the residual — a pattern consistent with a systematic per-layer
> discrepancy rather than compounding numerics. The leading hypothesis is the
> reference's bf16 downcast of RoPE cos/sin (measured at max_abs 0.022), amplified
> through the 128-term qk dot product. A single-layer fp32 bisect settles it; results
> in `results/published/bisect_fp32.json`. Behaviour is unaffected: 48/48 on the
> agentic ladder.

Then replace it with the answer once you have it. "Unresolved, here's the experiment"
is a stronger position than "it's numerics" backed by a curve that says otherwise —
and if the RoPE hypothesis holds, the final version of this paragraph is the best
result in the project.

## What's still open

| item | state |
|---|---|
| `bisect --dtype fp32` (the decisive run) | **not run** — tool now exists |
| `--bf16-rope` hypothesis test | **not run** — tool now exists |
| README fidelity section | needs the correction in §6 |
| P4.5 suite widening (8 → ~30 cases) | not started |
| `--repeats 3` on the published ladder | flag works, unused |
| git commit + manifest re-stamp | not done |
| HF weight upload | prepared, `--dry-run` verified, not pushed |
