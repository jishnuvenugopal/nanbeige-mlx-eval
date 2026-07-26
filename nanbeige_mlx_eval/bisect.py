"""Single-layer, submodule-level divergence bisect — the decisive fidelity test.

Why this exists
---------------
``trace.py`` answers *where* the port and the reference diverge across depth.
It does not answer *why*, and its full-model fp32 mode is not runnable on 16 GB.
That has been cited as a blocker. It is not: the experiment that distinguishes
"logic bug" from "numerics" needs **one layer**, not the whole model.

One Nanbeige decoder layer is ~143 M params:

    q 3072x6144 + k,v 3072x1024 x2 + o 6144x3072
      + gate,up 3072x10752 x2 + down 10752x3072 + 2 norms
    = 143.1 M  ->  573 MB in fp32, ~1.2 GB for both frameworks together.

So the fp32 comparison that was declared impossible is in fact cheap.

What it does
------------
Feeds the **same** input hidden state to both frameworks and compares stage by
stage. Stages are evaluated **independently from a common input**, not chained,
so an early discrepancy cannot mask or manufacture a later one:

    1.  input_layernorm            (RMSNorm alone)
    2.  q_proj / k_proj / v_proj   (pure matmul)
    3.  RoPE applied to q, k       (the rotation itself)
    4.  attention probs @ v        (softmax path, pre-o_proj)
    5.  o_proj
    6.  post_attention_layernorm
    7.  gate_proj / up_proj / SwiGLU / down_proj
    8.  the whole block

The first stage that falls below ~0.9999 **in fp32** is the bug. If every stage
clears 0.9999 in fp32 but the block degrades in bf16, it is precision, and the
same ladder in bf16 tells you which operation loses it.

Hypothesis this was built to test
---------------------------------
The reference downcasts RoPE cos/sin to bf16 before applying them
(``modeling_nanbeige.py``: ``return cos.to(dtype=x.dtype)``), while
``mx.fast.rope`` stays in fp32. ``parity.rope_precision_report`` measured that
floor at **max_abs 0.022** on the cos values and dismissed it as small because
the *cosine* was 0.999997. But cosine is the wrong lens here: a ~2% elementwise
perturbation on **both** q and k is amplified by the 128-term dot product into
the attention logits, and softmax turns a ~1-unit logit shift into materially
different attention weights. That would produce exactly what the trace shows —
a roughly constant per-layer error, present at layer 0, device-independent
(hence unchanged by moving off MPS), and diluted as the residual stream grows.

``emulate_reference_rope_precision`` tests it directly: round cos/sin to bf16 on
the MLX side and re-measure. If agreement jumps, the port is *more* accurate
than the reference and the gap has a known, quantified cause.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _flat(x) -> list[float]:
    """Flatten an mx.array or torch.Tensor to a python float list (fp32)."""
    if isinstance(x, mx.array):
        return x.astype(mx.float32).reshape(-1).tolist()
    return x.detach().to("cpu").float().reshape(-1).tolist()


def compare(a, b) -> dict[str, float]:
    """Cosine / max-abs / relative-L2 between two same-shaped tensors."""
    av, bv = _flat(a), _flat(b)
    if len(av) != len(bv):
        raise ValueError(f"shape mismatch: {len(av)} vs {len(bv)}")
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av))
    nb = math.sqrt(sum(y * y for y in bv))
    diff = [x - y for x, y in zip(av, bv)]
    nd = math.sqrt(sum(d * d for d in diff))
    return {
        "cosine": round(dot / (na * nb + 1e-30), 8),
        "max_abs": round(max((abs(d) for d in diff), default=0.0), 6),
        "rel_l2": round(nd / (na + 1e-30), 8),
        "norm_ref": round(na, 4),
    }


def verdict(m: dict[str, float], dtype: str) -> str:
    """Interpret one stage's numbers for the dtype it was measured at."""
    c = m["cosine"]
    if dtype == "fp32":
        if c >= 0.99999:
            return "ok"
        if c >= 0.999:
            return "SUSPECT (fp32 should be ~1e-6)"
        return "BUG (fp32 disagreement is not numerics)"
    # bf16 expectations are looser but still tight per-op
    if c >= 0.999:
        return "ok"
    if c >= 0.99:
        return "SUSPECT"
    return "TOO LARGE FOR BF16"


# --------------------------------------------------------------------------
# weight loading — one layer only
# --------------------------------------------------------------------------

def _layer_keys(layer_idx: int) -> dict[str, str]:
    p = f"model.layers.{layer_idx}."
    return {
        short: p + short
        for short in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        )
    }


def load_layer_torch(src: Path, layer_idx: int, dtype: str):
    """Pull just this layer's tensors out of the shards, without loading the model."""
    import torch
    from safetensors import safe_open

    want = _layer_keys(layer_idx)
    td = torch.float32 if dtype == "fp32" else torch.bfloat16
    out: dict[str, Any] = {}
    remaining = set(want.values())
    for shard in sorted(src.glob("*.safetensors")):
        if not remaining:
            break
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for short, full in want.items():
                if full in keys and full in remaining:
                    out[short] = f.get_tensor(full).to(td)
                    remaining.discard(full)
    if remaining:
        raise KeyError(f"missing tensors for layer {layer_idx}: {sorted(remaining)}")
    return out


def load_layer_mlx(src: Path, layer_idx: int, dtype: str) -> dict[str, mx.array]:
    want = _layer_keys(layer_idx)
    md = mx.float32 if dtype == "fp32" else mx.bfloat16
    out: dict[str, mx.array] = {}
    remaining = set(want.values())
    for shard in sorted(src.glob("*.safetensors")):
        if not remaining:
            break
        w = mx.load(str(shard))
        for short, full in want.items():
            if full in w and full in remaining:
                out[short] = w[full].astype(md)
                remaining.discard(full)
        del w
    if remaining:
        raise KeyError(f"missing tensors for layer {layer_idx}: {sorted(remaining)}")
    return out


# --------------------------------------------------------------------------
# reference stages, mode 'real': the ACTUAL NanbeigeDecoderLayer
# --------------------------------------------------------------------------
#
# Why this mode exists
# --------------------
# ``_torch_stages`` below is a *reimplementation* of the reference layer. A
# bisect against it answers "does the MLX port agree with my mirror?", not
# "does the MLX port agree with Nanbeige's code?" -- any behaviour the real
# layer has that the mirror also lacks is invisible to it, by construction.
#
# That distinction turned out to matter. Two independent measurements against
# the real HF model (trace: layer-0 cosine 0.925; parity: final-logit cosine
# 0.847) disagree with the mirror-based bisect (~0.99999) by four orders of
# magnitude. When one measurement is the odd one out and it is also the only
# one not using the reference implementation, that measurement is the suspect.
#
# This mode instantiates ``modeling_nanbeige.NanbeigeDecoderLayer`` directly,
# loads the same weights, and hooks its submodules. It is the arbiter:
#
#   real-layer vs MLX ~= 0.925  -> the divergence is genuine and lives in the
#                                  layer; diff the mirror against the real
#                                  layer stage-by-stage to localise it.
#   real-layer vs MLX ~= 0.99999 -> the layer is fine against real code, and
#                                  the gap only appears in the full-model path
#                                  (embedding, mask, positions, loop/norm
#                                  structure) -- which trace exercises and a
#                                  single-layer bisect does not.

_HOOK_STAGES = {
    "input_layernorm": ("input_layernorm", "out"),
    "q_proj": ("self_attn.q_proj", "out"),
    "k_proj": ("self_attn.k_proj", "out"),
    "v_proj": ("self_attn.v_proj", "out"),
    "attn_out": ("self_attn.o_proj", "in"),      # o_proj's input == attn output
    "o_proj": ("self_attn.o_proj", "out"),
    "post_attention_layernorm": ("post_attention_layernorm", "out"),
    "gate_proj": ("mlp.gate_proj", "out"),
    "up_proj": ("mlp.up_proj", "out"),
    "swiglu": ("mlp.down_proj", "in"),           # down_proj's input == swiglu
    "down_proj": ("mlp.down_proj", "out"),
}


def _load_reference_module(src: Path):
    """Import the checkpoint's own modeling_nanbeige.py (no HF model load).

    The remote-code files use package-relative imports (``from .configuration_nanbeige
    import NanbeigeConfig``), so they must be loaded under a synthetic package
    rather than as standalone modules -- otherwise the relative import fails with
    "attempted relative import with no known parent package".
    """
    import importlib.util
    import sys
    import types

    for cand in (src, Path("reference")):
        mod_path = cand / "modeling_nanbeige.py"
        cfg_path = cand / "configuration_nanbeige.py"
        if not mod_path.exists():
            continue

        pkg_name = "_nanbeige_refcode"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(cand.resolve())]
            sys.modules[pkg_name] = pkg
        else:
            pkg = sys.modules[pkg_name]

        # load configuration_nanbeige first so modeling_nanbeige's relative import resolves
        if cfg_path.exists():
            spec_cfg = importlib.util.spec_from_file_location(
                f"{pkg_name}.configuration_nanbeige", cfg_path
            )
            mod_cfg = importlib.util.module_from_spec(spec_cfg)
            sys.modules[f"{pkg_name}.configuration_nanbeige"] = mod_cfg
            spec_cfg.loader.exec_module(mod_cfg)

        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.modeling_nanbeige", mod_path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{pkg_name}.modeling_nanbeige"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise FileNotFoundError(
        f"modeling_nanbeige.py not found in {src} or ./reference — "
        "--reference real needs the checkpoint's remote-code files."
    )


def _real_layer_stages(w, cfg: dict, x, dtype: str, layer_idx: int, src: Path):
    """Run the genuine NanbeigeDecoderLayer and capture its submodule outputs.

    Convention: **a differential test must compare against the artifact under
    dispute.** If the reference side is code you wrote (a "mirror"), the test
    can only find bugs you didn't also make in the mirror -- a shared misreading
    of the source propagates into both sides and cancels. This is why
    ``--reference real`` exists: it makes the checkpoint's own layer the arbiter
    rather than ``_torch_stages`` (a reimplementation). Record which side was
    used in every report (see the ``reference`` / ``reference_note`` fields).
    """
    import torch
    from transformers import AutoConfig  # type: ignore

    mod = _load_reference_module(src)
    cfg_obj = AutoConfig.from_pretrained(str(src), trust_remote_code=True)
    cfg_obj.rope_scaling = None          # same reason as parity.py / trace.py
    cfg_obj._attn_implementation = "eager"

    td = torch.float32 if dtype == "fp32" else torch.bfloat16
    layer = mod.NanbeigeDecoderLayer(cfg_obj, layer_idx).to(td).eval()
    missing, unexpected = layer.load_state_dict(
        {k: v.to(td) for k, v in w.items()}, strict=False
    )
    real_missing = [k for k in missing if "rotary_emb" not in k]
    if real_missing or unexpected:
        raise RuntimeError(
            f"real-layer weight load mismatch: missing={real_missing} "
            f"unexpected={list(unexpected)}"
        )

    caught: dict[str, Any] = {}

    def mk(name, which):
        def hook(_m, inp, out):
            t = inp[0] if which == "in" else (out[0] if isinstance(out, tuple) else out)
            caught[name] = t.detach().float()
        return hook

    handles = []
    for stage, (path, which) in _HOOK_STAGES.items():
        target = layer
        for part in path.split("."):
            target = getattr(target, part)
        handles.append(target.register_forward_hook(mk(stage, which)))

    B, L, _ = x.shape
    neg = torch.finfo(td).min
    causal = torch.full((L, L), neg, dtype=td).triu(1)[None, None]
    pos = torch.arange(L)[None]
    try:
        with torch.no_grad():
            out = layer(
                hidden_states=x.to(td),
                attention_mask=causal,
                position_ids=pos,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=torch.arange(L),
            )
    finally:
        for h in handles:
            h.remove()

    caught["block"] = (out[0] if isinstance(out, tuple) else out).detach().float()
    return caught


# --------------------------------------------------------------------------
# reference stages (torch), REIMPLEMENTING modeling_nanbeige.NanbeigeAttention
#
# NOTE: this is a mirror, not the reference. See the comment above
# `_real_layer_stages` for why that distinction is load-bearing. Prefer
# `--reference real` for any claim about agreement with Nanbeige's code.
# --------------------------------------------------------------------------

def _torch_stages(w, cfg: dict, x, dtype: str, bf16_rope: bool):
    """Return {stage_name: tensor} for the reference path.

    ``bf16_rope`` reproduces the reference's ``cos.to(x.dtype)`` downcast even
    when running the rest of the layer in fp32, which is what isolates the RoPE
    precision hypothesis.
    """
    import torch

    H = cfg["hidden_size"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps, theta = cfg["rms_norm_eps"], float(cfg["rope_theta"])
    B, L, _ = x.shape
    s: dict[str, Any] = {}

    def rms(t, weight):
        dt = t.dtype
        t32 = t.to(torch.float32)
        v = t32.pow(2).mean(-1, keepdim=True)
        return (weight * (t32 * torch.rsqrt(v + eps)).to(dt))

    def rotate_half(t):
        h = t.shape[-1] // 2
        return torch.cat((-t[..., h:], t[..., :h]), dim=-1)

    s["input_layernorm"] = xn = rms(x, w["input_layernorm.weight"])

    q = torch.nn.functional.linear(xn, w["self_attn.q_proj.weight"])
    k = torch.nn.functional.linear(xn, w["self_attn.k_proj.weight"])
    v = torch.nn.functional.linear(xn, w["self_attn.v_proj.weight"])
    s["q_proj"], s["k_proj"], s["v_proj"] = q, k, v

    q = q.view(B, L, nh, hd).transpose(1, 2)
    k = k.view(B, L, nkv, hd).transpose(1, 2)
    v = v.view(B, L, nkv, hd).transpose(1, 2)

    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
    pos = torch.arange(L, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    if bf16_rope:  # the reference's own downcast
        cos = cos.to(torch.bfloat16).to(torch.float32)
        sin = sin.to(torch.bfloat16).to(torch.float32)
    cos, sin = cos.to(q.dtype)[None, None], sin.to(q.dtype)[None, None]

    qr = q * cos + rotate_half(q) * sin
    kr = k * cos + rotate_half(k) * sin
    s["rope_q"], s["rope_k"] = qr, kr

    kr_e = kr.repeat_interleave(nh // nkv, dim=1)
    v_e = v.repeat_interleave(nh // nkv, dim=1)
    logits = torch.matmul(qr, kr_e.transpose(2, 3)) / math.sqrt(hd)
    causal = torch.full((L, L), torch.finfo(torch.float32).min).triu(1).to(logits.dtype)
    logits = logits + causal
    probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(qr.dtype)
    s["attn_logits"] = logits
    ao = torch.matmul(probs, v_e).transpose(1, 2).reshape(B, L, nh * hd)
    s["attn_out"] = ao

    s["o_proj"] = op = torch.nn.functional.linear(ao, w["self_attn.o_proj.weight"])
    h = x + op
    s["post_attention_layernorm"] = hn = rms(h, w["post_attention_layernorm.weight"])
    g = torch.nn.functional.linear(hn, w["mlp.gate_proj.weight"])
    u = torch.nn.functional.linear(hn, w["mlp.up_proj.weight"])
    s["gate_proj"], s["up_proj"] = g, u
    s["swiglu"] = sw = torch.nn.functional.silu(g) * u
    s["down_proj"] = dp = torch.nn.functional.linear(sw, w["mlp.down_proj.weight"])
    s["block"] = h + dp
    return s


# --------------------------------------------------------------------------
# port stages (mlx), calling the real mlx_nanbeige modules
# --------------------------------------------------------------------------

def _mlx_stages(w, cfg: dict, x: mx.array, dtype: str, bf16_rope: bool):
    from mlx_nanbeige.model import ModelArgs, TransformerBlock

    args = ModelArgs.from_dict(cfg)
    blk = TransformerBlock(args)
    blk.load_weights([(k, v) for k, v in w.items()], strict=True)
    md = mx.float32 if dtype == "fp32" else mx.bfloat16
    blk.set_dtype(md)
    mx.eval(blk.parameters())

    B, L, _ = x.shape
    nh, nkv, hd = args.num_attention_heads, args.num_key_value_heads, args.head_dim
    a = blk.self_attn
    s: dict[str, Any] = {}

    s["input_layernorm"] = xn = blk.input_layernorm(x)
    s["q_proj"], s["k_proj"], s["v_proj"] = a.q_proj(xn), a.k_proj(xn), a.v_proj(xn)

    q = s["q_proj"].reshape(B, L, nh, hd).transpose(0, 2, 1, 3)
    k = s["k_proj"].reshape(B, L, nkv, hd).transpose(0, 2, 1, 3)
    v = s["v_proj"].reshape(B, L, nkv, hd).transpose(0, 2, 1, 3)

    if bf16_rope:
        qr, kr = _rope_bf16_emulated(q, hd, float(args.rope_theta)), \
                 _rope_bf16_emulated(k, hd, float(args.rope_theta))
    else:
        qr, kr = a.rope(q), a.rope(k)
    s["rope_q"], s["rope_k"] = qr, kr

    from mlx_nanbeige.model import scaled_dot_product_attention
    ao = scaled_dot_product_attention(qr, kr, v, cache=None, scale=a.scale, mask="causal")
    s["attn_out"] = ao.transpose(0, 2, 1, 3).reshape(B, L, -1)
    s["o_proj"] = op = a.o_proj(s["attn_out"])

    h = x + op
    s["post_attention_layernorm"] = hn = blk.post_attention_layernorm(h)
    s["gate_proj"], s["up_proj"] = blk.mlp.gate_proj(hn), blk.mlp.up_proj(hn)
    s["swiglu"] = sw = nn.silu(s["gate_proj"]) * s["up_proj"]
    s["down_proj"] = dp = blk.mlp.down_proj(sw)
    s["block"] = h + dp
    mx.eval(list(s.values()))
    return s


def _rope_bf16_emulated(t: mx.array, head_dim: int, theta: float) -> mx.array:
    """Apply rotate-half RoPE with cos/sin rounded to bf16, as the reference does."""
    L = t.shape[2]
    inv = 1.0 / (theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    freqs = mx.arange(L).astype(mx.float32)[:, None] * inv[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos = mx.cos(emb).astype(mx.bfloat16).astype(t.dtype)[None, None]
    sin = mx.sin(emb).astype(mx.bfloat16).astype(t.dtype)[None, None]
    half = head_dim // 2
    rot = mx.concatenate([-t[..., half:], t[..., :half]], axis=-1)
    return t * cos + rot * sin


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

STAGE_ORDER = [
    "input_layernorm", "q_proj", "k_proj", "v_proj", "rope_q", "rope_k",
    "attn_out", "o_proj", "post_attention_layernorm",
    "gate_proj", "up_proj", "swiglu", "down_proj", "block",
]

# The residual stream at layer 0 is the raw embedding lookup, whose rows have
# RMS ~= 0.024 for this checkpoint (initializer_range 0.02). A unit-variance
# random probe is therefore ~42x too large and exercises a completely different
# numerical regime -- notably, `eps=1e-5` is 0.001% of the variance at RMS 1.0
# but 1.7% of it at RMS 0.024. Always state the regime a bisect was run in.
EMBED_RMS = 0.024


def make_input(
    src: Path, cfg: dict, *, mode: str, seq_len: int, seed: int,
    target_rms: float | None = None, prompt: str | None = None,
):
    """Build the probe hidden state. Returns (x_np, description).

    mode='real'    : the actual embed_tokens output for `prompt` -- the only
                     regime that corresponds to what layer 0 really sees.
    mode='scaled'  : random, rescaled to `target_rms` (for a scale sweep).
    mode='random'  : unit-variance random. Fine for catching logic bugs, which
                     are input-independent; misleading for precision claims.

    Convention: precision measurements use real activations; only logic checks
    may use synthetic input. (This codebase was bitten twice in one week by a
    random probe standing in for a real activation -- bisect.py and parity.py.)
    """
    import numpy as np

    if mode == "real":
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
        ids = tok.encode(prompt or "The capital of France is", add_special_tokens=False)
        emb = _embed_rows(src, ids)
        x = emb[None, :, :].astype("float32")
        rms = float(np.sqrt((x.astype("float64") ** 2).mean()))
        return x, f"real embed_tokens({len(ids)} tokens), rms={rms:.5f}"

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, seq_len, cfg["hidden_size"])).astype("float32")
    if mode == "scaled":
        tr = target_rms if target_rms is not None else EMBED_RMS
        x = (x / float(np.sqrt((x.astype("float64") ** 2).mean())) * tr).astype("float32")
        return x, f"random scaled to rms={tr:.5f}"
    return x, "random, rms=1.0 (NOT the real layer-0 regime)"


def _embed_rows(src: Path, ids: list[int]):
    """Read just the needed embedding rows straight out of the shards."""
    import json as _json
    import struct

    import numpy as np

    name = "model.embed_tokens.weight"
    for shard in sorted(src.glob("*.safetensors")):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = _json.loads(f.read(n))
        if name not in hdr:
            continue
        meta = hdr[name]
        if meta["dtype"] != "BF16":
            raise NotImplementedError(f"embed dtype {meta['dtype']}")
        off = 8 + n + meta["data_offsets"][0]
        mm = np.memmap(shard, dtype=np.uint16, mode="r", offset=off,
                       shape=tuple(meta["shape"]))
        u = np.stack([mm[i].astype(np.uint32) << 16 for i in ids])
        return u.view(np.float32)
    raise KeyError(name)


def run_bisect(
    src_dir: str | Path,
    *,
    layer_idx: int = 0,
    dtype: str = "fp32",
    seq_len: int = 8,
    seed: int = 0,
    bf16_rope: bool = False,
    output: str | Path | None = None,
    input_mode: str = "real",
    target_rms: float | None = None,
    prompt: str | None = None,
    reference: str = "real",
) -> dict[str, Any]:
    """Compare one decoder layer stage-by-stage between the port and the reference.

    ``dtype='fp32'`` is the decisive run for *logic*: ~573 MB per side, so it
    fits where the full-model fp32 comparison does not. Logic bugs are
    input-independent, so any probe finds them.

    ``dtype='bf16'`` results are only meaningful at the **real input scale**.
    Layer 0 sees the raw embedding (RMS ~0.024 here); a unit-variance probe is
    ~42x too large and will report agreement the model never actually achieves.
    Hence ``input_mode`` defaults to ``'real'``.

    ``bf16_rope=True`` makes *both* sides use the reference's bf16 cos/sin
    downcast, isolating that one effect.
    """
    import numpy as np
    import torch

    src = Path(src_dir)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))

    x_np, input_desc = make_input(
        src, cfg, mode=input_mode, seq_len=seq_len, seed=seed,
        target_rms=target_rms, prompt=prompt,
    )
    seq_len = x_np.shape[1]
    input_rms = float(np.sqrt((x_np.astype("float64") ** 2).mean()))

    tw = load_layer_torch(src, layer_idx, dtype)
    tx = torch.from_numpy(x_np).to(torch.float32 if dtype == "fp32" else torch.bfloat16)
    if reference == "real":
        if bf16_rope:
            raise ValueError("--bf16-rope only applies to --reference mirror")
        ts = _real_layer_stages(tw, cfg, tx, dtype, layer_idx, src)
    else:
        ts = _torch_stages(tw, cfg, tx, dtype, bf16_rope)
        ts = {k: v.detach().float() for k, v in ts.items()}
    del tw

    mw = load_layer_mlx(src, layer_idx, dtype)
    mxx = mx.array(x_np).astype(mx.float32 if dtype == "fp32" else mx.bfloat16)
    ms = _mlx_stages(mw, cfg, mxx, dtype, bf16_rope)
    del mw

    stages = []
    first_bad = None
    for name in STAGE_ORDER:
        if name not in ts or name not in ms:
            continue
        m = compare(ts[name], ms[name])
        v = verdict(m, dtype)
        stages.append({"stage": name, **m, "verdict": v})
        if first_bad is None and v != "ok":
            first_bad = name

    report = {
        "src": str(src),
        "layer_idx": layer_idx,
        "dtype": dtype,
        "seq_len": seq_len,
        "seed": seed,
        "input_mode": input_mode,
        "reference": reference,
        "reference_note": (
            "genuine modeling_nanbeige.NanbeigeDecoderLayer"
            if reference == "real"
            else "REIMPLEMENTATION (_torch_stages) — agreement here does NOT "
                 "establish agreement with Nanbeige's own code"
        ),
        "input_desc": input_desc,
        "input_rms": round(input_rms, 6),
        "bf16_rope_on_both_sides": bf16_rope,
        "first_divergent_stage": first_bad,
        "block_cosine": next(s["cosine"] for s in stages if s["stage"] == "block"),
        "stages": stages,
        "interpretation": _interpret(first_bad, dtype, bf16_rope),
    }
    if dtype == "bf16" and input_mode == "random":
        report["WARNING"] = (
            f"Measured at input rms={input_rms:.3f}, but layer 0 really sees "
            f"rms~{EMBED_RMS}. bf16 agreement is strongly scale-dependent here; "
            f"do not cite this number as the per-layer precision floor. "
            f"Re-run with --input real."
        )
    if output:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _interpret(first_bad: str | None, dtype: str, bf16_rope: bool) -> str:
    if first_bad is None:
        if dtype == "fp32":
            return (
                "Every stage agrees to fp32 precision: the port's arithmetic is "
                "correct and the observed logit gap is a precision effect. Re-run "
                "with --dtype bf16 to find which operation loses it."
            )
        return "Every stage agrees within bf16 tolerance for this layer."
    if dtype == "fp32":
        return (
            f"'{first_bad}' disagrees in fp32. That is a logic error, not numerics — "
            f"the inputs to that stage matched and its output did not. Fix it before "
            f"drawing any conclusion from the full-model parity number."
        )
    return (
        f"'{first_bad}' is the first stage to lose agreement at bf16"
        + (" (with the reference's bf16 RoPE downcast applied to both sides)"
           if bf16_rope else "")
        + ". Re-run with --dtype fp32: if it is clean there, this is precision, not logic."
    )


def run_scale_sweep(
    src_dir: str | Path,
    *,
    layer_idx: int = 0,
    dtype: str = "bf16",
    seq_len: int = 8,
    seed: int = 0,
    rms_values: list[float] | None = None,
    reference: str = "real",
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Re-run the bisect across input magnitudes; report block cosine vs RMS.

    This is the experiment that reconciles ``bisect`` with ``trace``. If block
    agreement collapses as the input RMS falls toward the real embedding scale
    (~0.024), the divergence is a small-magnitude precision effect and the
    stage table at the *low* end names where it happens. If agreement is flat,
    the two tools disagree for some other reason and ``trace`` is the suspect.
    """
    rms_values = rms_values or [1.0, 0.5, 0.2, 0.1, 0.05, EMBED_RMS, 0.01]
    rows = []
    for r in rms_values:
        rep = run_bisect(
            src_dir, layer_idx=layer_idx, dtype=dtype, seq_len=seq_len,
            seed=seed, input_mode="scaled", target_rms=r, reference=reference,
        )
        worst = min(
            (s for s in rep["stages"] if s["stage"] != "block"),
            key=lambda s: s["cosine"],
        )
        rows.append({
            "input_rms": r,
            "block_cosine": rep["block_cosine"],
            "one_minus_block": round(1.0 - rep["block_cosine"], 9),
            "worst_stage": worst["stage"],
            "worst_stage_cosine": worst["cosine"],
        })

    real = run_bisect(
        src_dir, layer_idx=layer_idx, dtype=dtype, seq_len=seq_len,
        seed=seed, input_mode="real", reference=reference,
    )
    out = {
        "layer_idx": layer_idx,
        "dtype": dtype,
        "reference": reference,
        "reference_note": real["reference_note"],
        "sweep": rows,
        "real_embedding": {
            "input_rms": real["input_rms"],
            "block_cosine": real["block_cosine"],
            "stages": real["stages"],
        },
        "interpretation": _interpret_sweep(rows, real),
    }
    if output:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _interpret_sweep(rows: list[dict], real: dict[str, Any]) -> str:
    hi = rows[0]["one_minus_block"]
    lo = rows[-1]["one_minus_block"]
    ratio = (lo / hi) if hi > 0 else float("inf")
    real_gap = 1.0 - real["block_cosine"]
    base = (
        f"1-cosine grows {ratio:.0f}x going from input rms=1.0 to "
        f"rms={rows[-1]['input_rms']}. At the real embedding scale "
        f"(rms={real['input_rms']}) the block gap is {real_gap:.3e}. "
    )
    if real_gap > 1e-3:
        return base + (
            "This reproduces the trace's layer-0 divergence and confirms it is a "
            "small-magnitude precision effect, not a logic error. The lowest-cosine "
            "stage in `real_embedding.stages` is where it originates -- fix or "
            "upcast that operation."
        )
    if ratio < 5:
        return base + (
            "Agreement is essentially flat in input scale, so scale does NOT explain "
            "the gap between this bisect (~1e-5) and the trace's layer-0 cosine of "
            "0.925. Before blaming trace.py, check WHAT THIS BISECT COMPARED AGAINST: "
            "if `reference` is 'mirror', it validated the port against a "
            "reimplementation, not against Nanbeige's code, and cannot rule out a "
            "real divergence. Re-run with --reference real. Only if the real "
            "NanbeigeDecoderLayer also agrees at ~1e-5 is trace.py the suspect."
        )
    return base + (
        "Scale dependence is real but does not by itself reach the trace's layer-0 "
        "gap. Both effects are in play; widen the sweep and cross-check trace.py."
    )


def render_sweep_markdown(out: dict[str, Any]) -> str:
    lines = [
        f"# Input-scale sweep — layer {out['layer_idx']}, {out['dtype']}\n\n",
        "| input rms | block cosine | 1-cosine | worst stage | that stage |\n",
        "|---|---|---|---|---|\n",
    ]
    for r in out["sweep"]:
        lines.append(
            f"| {r['input_rms']} | {r['block_cosine']} | {r['one_minus_block']:.3e} "
            f"| {r['worst_stage']} | {r['worst_stage_cosine']} |\n"
        )
    re_ = out["real_embedding"]
    lines.append(
        f"| **real embed ({re_['input_rms']})** | **{re_['block_cosine']}** | "
        f"**{1 - re_['block_cosine']:.3e}** | — | — |\n\n"
    )
    lines.append(f"{out['interpretation']}\n")
    return "".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Single-layer bisect — layer {report['layer_idx']}, {report['dtype']}\n\n",
        f"- input: {report.get('input_desc', 'n/a')} "
        f"(rms={report.get('input_rms', 'n/a')})\n",
        f"- bf16 RoPE emulated on both sides: `{report['bf16_rope_on_both_sides']}`\n",
        f"- reference: `{report.get('reference', 'n/a')}` — {report.get('reference_note', '')}\n",
    ]
    if report.get("WARNING"):
        lines.append(f"\n> **WARNING** {report['WARNING']}\n\n")
    lines += [
        f"- first divergent stage: **{report['first_divergent_stage'] or 'none'}**\n",
        f"- block cosine: **{report['block_cosine']}**\n\n",
        "| stage | cosine | max_abs | rel_l2 | verdict |\n|---|---|---|---|---|\n",
    ]
    for s in report["stages"]:
        lines.append(
            f"| {s['stage']} | {s['cosine']} | {s['max_abs']} | {s['rel_l2']} | {s['verdict']} |\n"
        )
    lines.append(f"\n{report['interpretation']}\n")
    return "".join(lines)
