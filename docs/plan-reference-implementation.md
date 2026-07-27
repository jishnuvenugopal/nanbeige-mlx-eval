# Plan — a vendored, reproducible `NanbeigeForCausalLM`

Status: proposed, not started. Written 2026-07-26.

---

## The problem

Every fidelity claim in this project is measured against a reference that the
project does not contain and cannot pin.

| property of the current reference | consequence |
|---|---|
| `reference/` is **gitignored** | nobody who clones the repo can run `parity`, `trace`, `bisect` or `crosscheck` at all |
| loaded via `trust_remote_code` from the Hub | Nanbeige can update `modeling_nanbeige.py` at any time; measurements would silently drift |
| 2671 lines, written against transformers 4.42, run under 5.14.1 | needs `cfg.rope_scaling = None` in 4 files, plus `_hfquiet.py` to suppress `use_return_dict` deprecation noise |
| ~85% dead code for this config | n-gram embeddings, hyper-connection, MHC, depth attention, double-loop split, `loop_share_kv`, `qk_layernorm` — all disabled, all still executed as branch checks |
| requires the 8 GB checkpoint to instantiate | nothing fidelity-related can run in CI |

So the most important artifact in the repo — the thing that decides whether the
port is correct — is an unpinned download that only works on one machine with
one transformers version. That is the opposite of reproducible.

## The proposal

Vendor `nanbeige_ref/` — a minimal, self-contained PyTorch implementation of
`NanbeigeForCausalLM` covering exactly the code path the 4.2 checkpoint uses.
Roughly 7 classes and ~300 lines against the upstream's 17 and 2671:

```
NanbeigeRMSNorm          NanbeigeRotaryEmbedding    NanbeigeMLP
NanbeigeAttention        NanbeigeDecoderLayer
NanbeigeModel            NanbeigeForCausalLM
```

Depends on `torch` only. No `transformers` import for the model itself (the
tokenizer keeps using `AutoTokenizer`, which is stable and separable). No
`PreTrainedModel`, no `Cache` classes, no `_update_causal_mask` boilerplate — a
plain `nn.Module` with a `forward` that takes `input_ids` and returns logits.

---

## The trap this must not fall into

**A reference written by reading `modeling_nanbeige.py` can inherit the same
misreading as the port and cancel it out.** That is precisely what happened with
`_torch_stages` in `bisect.py`: it agreed with the MLX port at cosine 0.99999
while both disagreed with the real layer at 0.925, and the bisect was blind to it
by construction (Addendum 3).

So the minimal reference is **worthless until proven equivalent to the vendored
upstream**, and that proof must come before any tool is switched over to it.
Phase 2 is not a formality; it is the whole plan.

The rule already in the codebase applies here verbatim:

> A differential test must compare against the artifact under dispute. If the
> reference side is code you wrote, the test can only find bugs you didn't also
> make there.

The minimal reference earns the right to be a reference only by matching the
thing it replaces, on real weights, at the full-model level.

---

## Phase 1 — write it

**Location:** `nanbeige_ref/` in the eval repo. It is validation apparatus, not
a runtime dependency of the port, so it does not belong in `nanbeige-mlx`.

**Fidelity to upstream, not elegance.** Mirror the upstream's op *order* exactly
where it is observable, even where a cleaner formulation exists. Specifically:

- `NanbeigeRMSNorm` upcasts to fp32 for the variance, applies `weight` after
  casting back (`modeling_nanbeige.py:379-386`).
- `NanbeigeRotaryEmbedding` builds `inv_freq` in fp32, computes `cos`/`sin` in
  fp32 under `autocast(enabled=False)`, then **downcasts to `x.dtype`**
  (`:952-965`). Keep the downcast — it is a real behaviour, not a bug to fix.
- `inv_freq` must be a **non-persistent buffer** and must survive `.to(dtype)`.
  Hold it as fp32 explicitly and cast at use, so a blanket `.to(bfloat16)`
  cannot degrade it — the failure that produced the 0.925 in `bisect.py`.
- Eager attention only: `matmul(q, kᵀ)/sqrt(head_dim)` + additive mask, softmax
  in fp32 cast back (`:1194-1200`), `repeat_kv` grouping `h // n_rep`.
- The loop: `for loop_idx in range(num_loops)` over all layers, `self.norm`
  applied at each boundary when `skip_loop_final_norm` is False (`:2217-2308`).
- KV cache index `layer_idx + loop_idx * num_hidden_layers` when caching.

**Explicitly out of scope** — raise `NotImplementedError` if the config asks for
them, so the reference can never silently diverge by ignoring a flag:
`enable_hyper_connection`, `enable_mhc`, `enable_depth_attention`,
`enable_double_loop_split`, `loop_share_kv`, `qk_layernorm`, n-gram embeddings,
`pretraining_tp > 1`, `rope_scaling` non-null.

**Config:** a dataclass read from `config.json`, not a `PretrainedConfig`
subclass. Assert the disabled-feature flags on construction.

**Weights:** load directly from safetensors by key. The key layout is already
known to map 1:1 (12 unique patterns, 201 tensors).

---

## Phase 2 — the equivalence gate

The make-or-break step. New command:

```bash
nanbeige-mlx-eval refcheck --src models/nanbeige42-hf --dtype bf16 --gate
```

Runs the six `parity` prompts through **vendored upstream** and **minimal
reference** on CPU with the same weights, and compares:

1. **Final logits** — the bar. Same dtype, same device, same op order should give
   near-bit-identical results.
2. **All 44 effective-layer hidden states** — reuse `trace.py`'s forward-hook
   machinery on the upstream side and the equivalent capture on the minimal side.
   A single divergent layer localises the misreading immediately.
3. **Per-stage within layer 0** — reuse `bisect.py`'s `_HOOK_STAGES` table.

**Pass bar (bf16, CPU):** logit cosine ≥ 0.999999 **and** max-abs logit
difference < 1e-3 on all six prompts, **and** every one of the 44 layer states
above cosine 0.99999.

That bar is deliberately far tighter than the 0.99 fidelity gate. These are two
PyTorch implementations of the same arithmetic on the same device — anything
looser means a real difference, and the correct response is to find it, not to
relax the threshold. **If it fails, Phase 3 does not start.**

Commit the passing artifact to `results/published/refcheck.json` so the
equivalence claim is auditable rather than asserted.

---

## Phase 3 — switch the tools over

Add `--reference {upstream,minimal}` to `parity`, `trace`, `bisect` and
`crosscheck`, defaulting to `minimal` once Phase 2 passes. Keep `upstream`
working — it is how Phase 2 is re-run when the checkpoint or transformers moves.

Record which was used in every artifact, same as `bisect`'s existing
`reference` / `reference_note` fields. An artifact that doesn't say what it
compared against is the problem Addendum 3 was about.

What this buys immediately:

- `reference/` comes **out of `.gitignore`** — or rather becomes unnecessary,
  since `nanbeige_ref/` is committed and versioned.
- `cfg.rope_scaling = None` and `_hfquiet.py` become upstream-path-only.
- Measurements stop depending on the Hub's current copy of the remote code.
- A reader can reproduce every fidelity number from a clone plus the weights.

---

## Phase 4 — a synthetic fixture, and fidelity tests in CI

The minimal reference has no `PreTrainedModel` machinery, so it can be
instantiated at **any** size. Build `tests/fixtures/tiny_nanbeige/`: a config
with `hidden_size=64, num_hidden_layers=2, num_loops=2, num_attention_heads=4,
num_key_value_heads=2, head_dim=16, vocab_size=128`, plus seeded random weights
in a ~2 MB safetensors file.

That makes the following runnable in CI **for the first time**:

- minimal reference vs MLX port, full forward, on a tiny model
- the 44-slot cache logic (here 4 slots) via the same prefill-vs-decode assertion
- `trace` / `bisect` / `crosscheck` end-to-end on a fixture

torch has Linux wheels, so the reference side runs on `ubuntu-latest`. The MLX
side still needs `macos-14` — but a structural regression in the reference, the
trace harness or the bisect harness would now be caught by CI instead of by a
person three rounds later. Given this project's history, that is the single
highest-value item in the plan after Phase 2.

---

## Phase 5 — the ablation that could close the 0.847

This is the part that makes the work worth more than hygiene.

Six hypotheses about the end-to-end gap have been proposed and falsified, each
by guessing a mechanism and testing it. A reference *you control* replaces
guessing with enumeration: the set of implementation choices where two faithful
implementations may legitimately differ is **small and listable**. Put each
behind a flag and toggle it one at a time.

Candidate flags, each a one-line change in `nanbeige_ref/`:

| flag | upstream behaviour | plausible MLX behaviour |
|---|---|---|
| `rmsnorm_accum` | fp32 variance, cast back | fused kernel, fp32 internal |
| `softmax_dtype` | fp32 then cast to q dtype | fp32 internal |
| `rope_trig_dtype` | fp32 then **downcast to x.dtype** | fp32 throughout |
| `attn_accum_dtype` | bf16 matmul, fp32 softmax | fp32 accumulate |
| `residual_order` | `x + attn`, then `h + mlp` | same, but fused |
| `swiglu_dtype` | bf16 elementwise | fp32 intermediate |
| `loop_norm_position` | norm after the 22nd block | same |
| `lm_head_dtype` | bf16 | bf16 |

Procedure: run full `parity` with each flag flipped individually (8 runs, minutes
each on CPU). If any single flip moves the cosine from 0.847 toward 0.99, that is
the answer, and it will be the first mechanistic explanation the project has that
was *found* rather than *guessed*. If none does, that itself is informative — it
says the difference is not in this enumerable set, and the next place to look is
the harness rather than the arithmetic.

**Discipline for this phase, given the record:** run the ablation first, write
the conclusion second. The scoreboard is seven hypotheses, seven wrong, every one
killed by a measurement that took minutes.

---

## Effort, sequencing, risk

| phase | effort | gates |
|---|---|---|
| 1 — write it | ~half a day | — |
| 2 — equivalence | ~half a day, mostly debugging | **gates 3, 4, 5** |
| 3 — switch tools | ~2 hours | Phase 2 passing |
| 4 — fixture + CI | ~half a day | Phase 1 (not 2 — the fixture is independent) |
| 5 — ablation | ~2 hours to run, unknown to interpret | Phase 3 |

**Main risk:** Phase 2 fails and the divergence is hard to localise. Mitigated by
the layered comparison — logits, then 44 layer states, then per-stage in layer 0.
The infrastructure to do all three already exists in `trace.py` and `bisect.py`;
this reuses it rather than building new instruments.

**Secondary risk:** scope creep into reimplementing the disabled feature zoo.
The `NotImplementedError` guards are the defence — the reference covers this
checkpoint and refuses anything else, by design.

**Not a goal:** replacing the upstream reference. It stays as the authority that
validates the minimal one. The minimal one is a *pinned, testable proxy* that
earns its status in Phase 2 and can be re-validated whenever either side moves.

---

## Relationship to the release

Independent of it. The release (PyPI, HF weights) is blocked only on credentials
and can proceed in parallel. Nothing in this plan changes the port, the quants,
or any published number — it changes what those numbers can be checked against.

If Phase 5 does explain the 0.847, the fidelity sections in both READMEs and the
model cards get rewritten one more time, and the investigation log gets its
closing addendum.
