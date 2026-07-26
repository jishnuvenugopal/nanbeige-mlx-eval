"""Cross-validate `bisect` against `trace` on the one layer where they disagree.

The contradiction
-----------------
Both tools now measure the same thing — layer 0's output at the last token, in
bf16, at the real embedding scale, against Nanbeige's own ``NanbeigeDecoderLayer``:

    bisect --dtype bf16 --input real --reference real   ->  0.99999326
    trace  effective layer 0                            ->  0.925086

Those cannot both be right. There are exactly four tensors in play:

    A = bisect, MLX side          B = bisect, HF side
    C = trace,  MLX side          D = trace,  HF side

We know ``cos(A,B) ~= 0.99999`` and ``cos(C,D) ~= 0.925``. Therefore **at least
one of cos(A,C) or cos(B,D) is far from 1** — the two tools are not feeding the
same thing to at least one framework. Computing the full 4x4 matrix names the
broken side in a single run, with no further hypothesising.

    cos(A,C) low  -> the MLX side differs between the tools
    cos(B,D) low  -> the HF side differs between the tools
    both high     -> arithmetically impossible; a slicing/ordering bug in how
                     one of the tools pairs its own two sides

The embedding is checked too
----------------------------
`bisect` hands **both** frameworks the same numpy array (read from the shards by
``_embed_rows``). `trace` lets **each framework do its own ``embed_tokens``
lookup**. So if the two lookups disagree, `bisect` is blind to it by
construction — the same class of blindness as comparing against a
reimplementation instead of the reference. That makes the embedding the most
upstream suspect, and it is nearly free to check, so it is step 0 here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = "The capital of France is"


def _cos(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-30)


def _maxabs(a: list[float], b: list[float]) -> float:
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def _rms(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a) / max(len(a), 1))


def _pair(name_a: str, a: list[float], name_b: str, b: list[float]) -> dict[str, Any]:
    return {
        "pair": f"{name_a} vs {name_b}",
        "cosine": round(_cos(a, b), 8),
        "max_abs": round(_maxabs(a, b), 6),
        "rms_a": round(_rms(a), 6),
        "rms_b": round(_rms(b), 6),
    }


# --------------------------------------------------------------------------
# step 0 — the embedding
# --------------------------------------------------------------------------

def _embeddings(src: Path, ids: list[int], dtype: str) -> dict[str, list[float]]:
    """Last-token embedding as produced three different ways."""
    import mlx.core as mx
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore

    from .bisect import _embed_rows

    md = mx.bfloat16 if dtype == "bf16" else mx.float32
    td = torch.bfloat16 if dtype == "bf16" else torch.float32

    # (i) raw shard read — what bisect hands to BOTH sides
    shard = _embed_rows(src, ids)                      # (L, H) fp32, lossless
    raw = shard[-1].astype("float32").tolist()

    # (ii) MLX's own embed_tokens lookup — what trace's MLX side uses.
    # NOTE: build a bare nn.Embedding, NOT Model(args). Instantiating the full
    # model randomly initialises ~4B parameters in fp32 (~16 GB) just to reach
    # one lookup table, which OOMs a 16 GB machine before it can tell you
    # anything.
    import mlx.nn as mlx_nn
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    emb_layer = mlx_nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
    for sh in sorted(src.glob("*.safetensors")):
        d = mx.load(str(sh))
        if "model.embed_tokens.weight" in d:
            emb_layer.load_weights(
                [("weight", d["model.embed_tokens.weight"].astype(md))]
            )
            del d
            break
        del d
    mlx_lookup = emb_layer(mx.array([ids]))[0, -1].astype(mx.float32).tolist()
    del emb_layer

    # (iii) HF's own embed_tokens lookup — what trace's HF side uses
    cfg_obj = AutoConfig.from_pretrained(str(src), trust_remote_code=True)
    cfg_obj.rope_scaling = None
    hf = AutoModelForCausalLM.from_pretrained(
        str(src), config=cfg_obj, trust_remote_code=True,
        dtype=td, attn_implementation="eager",
    ).eval()
    with torch.no_grad():
        hf_emb = hf.model.embed_tokens(torch.tensor([ids]))[0, -1]
    hf_lookup = hf_emb.float().tolist()
    del hf

    return {"shard_read": raw, "mlx_lookup": mlx_lookup, "hf_lookup": hf_lookup}


# --------------------------------------------------------------------------
# the four layer-0 tensors
# --------------------------------------------------------------------------

def _bisect_sides(src: Path, ids: list[int], dtype: str, prompt: str):
    """A = bisect MLX block output, B = bisect real-HF block output."""
    import mlx.core as mx
    import torch

    from .bisect import (
        _mlx_stages,
        _real_layer_stages,
        load_layer_mlx,
        load_layer_torch,
        make_input,
    )

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    x_np, _ = make_input(src, cfg, mode="real", seq_len=len(ids), seed=0,
                         prompt=prompt)

    tw = load_layer_torch(src, 0, dtype)
    tx = torch.from_numpy(x_np).to(torch.bfloat16 if dtype == "bf16" else torch.float32)
    B = _real_layer_stages(tw, cfg, tx, dtype, 0, src)["block"][0, -1].float().tolist()
    del tw

    mw = load_layer_mlx(src, 0, dtype)
    mxx = mx.array(x_np).astype(mx.bfloat16 if dtype == "bf16" else mx.float32)
    A = _mlx_stages(mw, cfg, mxx, dtype, False)["block"][0, -1]
    A = A.astype(mx.float32).tolist()
    del mw
    return A, B


def _trace_sides(src: Path, ids: list[int], dtype: str):
    """C = trace MLX layer-0 state, D = trace HF layer-0 state."""
    from .trace import _gather_hf_layer_states, _gather_mlx_layer_states

    D = _gather_hf_layer_states(src, ids, device="cpu", dtype=dtype)[0]
    C = _gather_mlx_layer_states(src, ids, dtype=dtype)[0]
    return C, D


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_crosscheck(
    src_dir: str | Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    dtype: str = "bf16",
    output: str | Path | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer  # type: ignore

    src = Path(src_dir)
    tok = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
    ids = tok.encode(prompt, add_special_tokens=False)

    emb = _embeddings(src, ids, dtype)
    emb_pairs = [
        _pair("shard_read", emb["shard_read"], "mlx_lookup", emb["mlx_lookup"]),
        _pair("shard_read", emb["shard_read"], "hf_lookup", emb["hf_lookup"]),
        _pair("mlx_lookup", emb["mlx_lookup"], "hf_lookup", emb["hf_lookup"]),
    ]

    A, B = _bisect_sides(src, ids, dtype, prompt)
    C, D = _trace_sides(src, ids, dtype)

    tensors = {"A_bisect_mlx": A, "B_bisect_hf": B, "C_trace_mlx": C, "D_trace_hf": D}
    names = list(tensors)
    matrix = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            matrix.append(_pair(na, tensors[na], nb, tensors[nb]))

    def get(pair: str) -> float:
        return next(m["cosine"] for m in matrix if m["pair"] == pair)

    ac = get("A_bisect_mlx vs C_trace_mlx")
    bd = get("B_bisect_hf vs D_trace_hf")
    ab = get("A_bisect_mlx vs B_bisect_hf")
    cd = get("C_trace_mlx vs D_trace_hf")

    out = {
        "src": str(src),
        "prompt": prompt,
        "n_tokens": len(ids),
        "dtype": dtype,
        "embedding_agreement": emb_pairs,
        "layer0_matrix": matrix,
        "key_numbers": {
            "within_bisect_A_vs_B": ab,
            "within_trace_C_vs_D": cd,
            "across_mlx_A_vs_C": ac,
            "across_hf_B_vs_D": bd,
        },
        "verdict": _verdict(ab, cd, ac, bd, emb_pairs),
    }
    if output:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _verdict(ab: float, cd: float, ac: float, bd: float, emb: list[dict]) -> str:
    emb_bad = [e["pair"] for e in emb if e["cosine"] < 0.9999]
    if emb_bad:
        return (
            f"EMBEDDING DISAGREES ({', '.join(emb_bad)}). This is upstream of every "
            "other measurement and explains the whole chain: bisect hands both sides "
            "the same array so it cannot see this, while trace lets each framework do "
            "its own lookup and therefore does. Fix the embedding path first; every "
            "downstream number is suspect until it agrees."
        )
    low = 0.999
    if ac < low and bd >= low:
        return (
            f"MLX SIDE DIFFERS between the tools (A vs C = {ac}). The bisect's "
            "TransformerBlock path and the trace's full-Model walk do not compute the "
            "same layer-0 output. Suspect the trace's MLX walk: mask construction "
            "(`create_attention_mask(h, cache[0])` with a None cache), the RoPE offset "
            "taken from a None cache, or `set_dtype` ordering relative to "
            "`load_weights`. The port itself is cleared -- the bisect's MLX path "
            "agrees with real HF."
        )
    if bd < low and ac >= low:
        return (
            f"HF SIDE DIFFERS between the tools (B vs D = {bd}). A standalone "
            "NanbeigeDecoderLayer and the same layer inside the full model produce "
            "different layer-0 output. That means the model-level forward does "
            "something to the hidden state before layer 0 that the standalone layer "
            "never sees, or the hook captures something other than the layer output. "
            "Read NanbeigeModel.forward between `inputs_embeds` and the layer loop, "
            "and confirm the hook's `out[0]` is the hidden state."
        )
    if ac < low and bd < low:
        return (
            f"BOTH sides differ between the tools (A/C={ac}, B/D={bd}). The two "
            "harnesses are set up differently in a way that hits both frameworks -- "
            "most likely the input (token ids, prompt, or which token is sliced). "
            "Check that both tools tokenise identically and both slice [0, -1]."
        )
    return (
        f"CONTRADICTION UNRESOLVED: A/B={ab}, C/D={cd}, A/C={ac}, B/D={bd}. If A~=C "
        "and B~=D but A~=B while C!~=D, the tensors are consistent and the fault is in "
        "how one tool PAIRS its own two sides -- check trace.py's layer ordering "
        "(hooks append in execution order; the MLX walk appends in loop-then-layer "
        "order) and confirm index 0 means the same layer on both sides."
    )


# --------------------------------------------------------------------------
# replay — which of B or D is the layer's true behaviour?
# --------------------------------------------------------------------------
#
# The crosscheck found cos(B,D) = 0.925: a standalone NanbeigeDecoderLayer and
# the same layer inside the full model disagree. It is tempting to read that as
# "the reference contradicts itself, so the port (which matches B) is fine."
# That reading is backwards:
#
#   * D is ground truth. It is what the model computes during a real forward
#     pass -- the thing that actually generates tokens. B is a layer invoked by
#     hand, outside its model, by the newest and least-tested code in the stack
#     (`_real_layer_stages`). Matching the artificial construction while missing
#     the real one is the opposite of exoneration.
#   * `parity` (0.847) compares full forward to full forward. No standalone
#     layer anywhere. The port is implicated by a measurement this crosscheck
#     does not touch.
#   * "HF contradicts itself" is the extraordinary claim. The ordinary one is
#     that `_real_layer_stages` invokes the layer differently than the model
#     does -- and if so, the bisect's 0.99999 never established anything,
#     because it validated the port against a mis-invocation the port may share.
#
# This replays the EXACT arguments the model passes to layer 0 into the
# standalone layer. It terminates the search:
#
#   replay matches D  -> the difference is in the ARGUMENTS. Diff them; the
#                        differing one is what the bisect got wrong, and very
#                        likely what the port gets wrong too.
#   replay matches B  -> the difference is in the layer OBJECT (attention class
#                        selected from config, or a config flag that differs
#                        between `from_pretrained(attn_implementation=...)` and
#                        a hand-set `cfg._attn_implementation`).

def _describe(v: Any) -> Any:
    import torch
    if isinstance(v, torch.Tensor):
        f = v.detach().float()
        return {
            "type": "Tensor", "shape": list(v.shape), "dtype": str(v.dtype),
            "min": round(f.min().item(), 6), "max": round(f.max().item(), 6),
            "mean": round(f.mean().item(), 6),
        }
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return f"<{type(v).__name__}>"


def run_replay(
    src_dir: str | Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    dtype: str = "bf16",
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Capture layer 0's real call arguments, replay them standalone, compare."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore

    from .bisect import _real_layer_stages, load_layer_torch

    src = Path(src_dir)
    td = torch.bfloat16 if dtype == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
    ids = tok.encode(prompt, add_special_tokens=False)

    cfg_obj = AutoConfig.from_pretrained(str(src), trust_remote_code=True)
    cfg_obj.rope_scaling = None
    model = AutoModelForCausalLM.from_pretrained(
        str(src), config=cfg_obj, trust_remote_code=True,
        dtype=td, attn_implementation="eager",
    ).eval()

    grabbed: dict[str, Any] = {}

    def pre_hook(_m, args, kwargs):
        if "args" not in grabbed:               # first call only == loop 0
            grabbed["args"] = args
            grabbed["kwargs"] = kwargs
        return None

    def post_hook(_m, _a, out):
        grabbed.setdefault("D", (out[0] if isinstance(out, tuple) else out).detach().float())

    layer0 = model.model.layers[0]
    h1 = layer0.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = layer0.register_forward_hook(post_hook)
    try:
        with torch.no_grad():
            model(torch.tensor([ids]), use_cache=False)
    finally:
        h1.remove(); h2.remove()

    # which attention class did the in-model layer actually get?
    inmodel_attn = type(layer0.self_attn).__name__

    # Replay those exact arguments into the SAME layer object.
    with torch.no_grad():
        replay_same_obj = layer0(*grabbed["args"], **grabbed["kwargs"])
    replay_same_obj = (
        replay_same_obj[0] if isinstance(replay_same_obj, tuple) else replay_same_obj
    ).detach().float()

    D = grabbed["D"]
    hs = grabbed["args"][0] if grabbed["args"] else grabbed["kwargs"]["hidden_states"]
    del model

    # Now the standalone layer the bisect builds, given the SAME hidden state.
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    tw = load_layer_torch(src, 0, dtype)
    B = _real_layer_stages(tw, cfg, hs.to(td), dtype, 0, src)["block"].detach().float()
    del tw

    d = D[0, -1].tolist()
    b = B[0, -1].tolist()
    r = replay_same_obj[0, -1].tolist()

    out_d = {
        "src": str(src), "prompt": prompt, "dtype": dtype, "n_tokens": len(ids),
        "inmodel_attention_class": inmodel_attn,
        "captured_call": {
            "positional": [_describe(a) for a in grabbed["args"]],
            "keyword": {k: _describe(v) for k, v in grabbed["kwargs"].items()},
        },
        "pairs": [
            _pair("D_inmodel", d, "B_standalone", b),
            _pair("D_inmodel", d, "R_replay_same_object", r),
            _pair("B_standalone", b, "R_replay_same_object", r),
        ],
        "verdict": _replay_verdict(_cos(d, r), _cos(d, b), inmodel_attn),
    }
    if output:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out_d, indent=2), encoding="utf-8")
    return out_d


def _replay_verdict(dr: float, db: float, attn: str) -> str:
    if dr > 0.9999 and db < 0.999:
        return (
            f"ARGUMENTS. Replaying the model's own call arguments into the same layer "
            f"reproduces D (cos={dr:.6f}), while the bisect's hand-built invocation does "
            f"not (cos={db:.6f}). So the layer is deterministic and correct; "
            f"`_real_layer_stages` calls it differently than the model does. Diff "
            f"`captured_call` against the bisect's invocation (mask construction, "
            f"cache_position, loop_idx, dtype) -- the differing argument is what the "
            f"bisect got wrong. CRITICAL: the port matches B, so it very likely shares "
            f"the same mistake. The bisect's 0.99999 does NOT clear the port."
        )
    if dr < 0.999:
        return (
            f"NON-DETERMINISM OR STATE. Replaying the model's exact arguments into the "
            f"same layer object did not reproduce its own output (cos={dr:.6f}). The "
            f"layer carries state across calls, or something outside it mutates the "
            f"hidden state. Look for in-place ops and for buffers on "
            f"NanbeigeRotaryEmbedding."
        )
    return (
        f"LAYER OBJECT. Same arguments give the same answer (cos={dr:.6f}) and the "
        f"standalone layer also agrees (cos={db:.6f}) -- so B and D should not have "
        f"differed. Re-check that the crosscheck fed B the same hidden state; the "
        f"in-model attention class here is `{attn}`, and if the bisect's standalone "
        f"layer selected a different class from a hand-set `_attn_implementation`, "
        f"that is the difference."
    )


def render_replay_markdown(out: dict[str, Any]) -> str:
    lines = [
        f"# Layer-0 argument replay — {out['dtype']}\n\n",
        f"in-model attention class: `{out['inmodel_attention_class']}`\n\n",
        "| pair | cosine | max_abs | rms A | rms B |\n|---|---|---|---|---|\n",
    ]
    for p in out["pairs"]:
        lines.append(
            f"| {p['pair']} | {p['cosine']} | {p['max_abs']} | {p['rms_a']} | {p['rms_b']} |\n"
        )
    lines.append("\n## Arguments the model actually passed\n\n```json\n")
    lines.append(json.dumps(out["captured_call"], indent=2))
    lines.append("\n```\n\n")
    lines.append(f"**Verdict.** {out['verdict']}\n")
    return "".join(lines)


def render_markdown(out: dict[str, Any]) -> str:
    lines = [
        f"# bisect vs trace cross-check — layer 0, {out['dtype']}\n\n",
        f"prompt: `{out['prompt']}` ({out['n_tokens']} tokens)\n\n",
        "## Step 0 — embedding (the one thing bisect cannot see)\n\n",
        "| pair | cosine | max_abs |\n|---|---|---|\n",
    ]
    for e in out["embedding_agreement"]:
        lines.append(f"| {e['pair']} | {e['cosine']} | {e['max_abs']} |\n")
    lines += [
        "\n## Layer-0 output, all pairs\n\n",
        "| pair | cosine | max_abs | rms A | rms B |\n|---|---|---|---|---|\n",
    ]
    for m in out["layer0_matrix"]:
        lines.append(
            f"| {m['pair']} | {m['cosine']} | {m['max_abs']} | {m['rms_a']} | {m['rms_b']} |\n"
        )
    lines.append(f"\n**Verdict.** {out['verdict']}\n")
    return "".join(lines)
