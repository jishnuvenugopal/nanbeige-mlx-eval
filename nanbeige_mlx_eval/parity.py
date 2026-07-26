"""Half A — fidelity gate: MLX port vs the HuggingFace reference.

A from-scratch MLX port of a custom architecture cannot be trusted on coherent
generation alone. This module runs the same prompts through:

* the official Nanbeige checkpoint under ``transformers`` (``trust_remote_code``),
  on a chosen device/dtype (default **CPU/bf16** — MPS bf16 is known-shaky and
  was the likely cause of the 0.844 cosine in the first version of this report),
  and
* this project's MLX port (loaded in memory from the source weights),

and compares next-token logits. The HF model is loaded, measured and freed
*before* the MLX model is loaded, so the two never co-reside (16 GB machine).

Because the two frameworks differ in matmul ordering and RMSNorm upcasting,
bit-exact equality is not expected; the bar is high *agreement* (top-1 identity
rate, logit cosine, KL). On CPU/bf16 the mean cosine should clear 0.99. The CLI
``--gate 0.99`` makes that bar an enforced threshold (exit non-zero if it fails).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_nanbeige.model import Model, ModelArgs


# Diverse probes: English, Chinese, code, and a tool-call prompt. Short prompts
# keep the comparison fast and exercise distinct token distributions.
DEFAULT_PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the factorial of n.",
    "中国的首都是",
    "Translate 'good morning' to French and Japanese.",
    "Sum the numbers 2, 4, 7, 9 and return only the total.",
    "雪化了之后会变成什么？",
]


def load_mlx_bf16(src_dir: str | Path) -> Any:
    """Build the MLX port in memory and load the source BF16 weights into it."""
    src = Path(src_dir)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    args = ModelArgs.from_dict(cfg)
    model = Model(args)

    weights: dict[str, mx.array] = {}
    for shard in sorted(src.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def _gather_hf_logits(
    src_dir: str | Path,
    prompts: list[str],
    *,
    device: str = "cpu",
    dtype: str = "bf16",
) -> tuple[list, list]:
    """Load HF model, return (input_ids_list, last_token_logits_list) in fp32.

    Defaults to **CPU/bf16**: MPS bf16 is known-shaky and was almost certainly
    the dominant cause of the 0.844 cosine the first version of this report
    published. bf16 on CPU is ~8.3 GB — fits a 16 GB machine.
    """
    import torch  # type: ignore
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

    # transformers 5.x auto-populates config.rope_scaling as {'rope_type':'default'}
    # for this checkpoint, but the model's (4.x-era) modeling code reads the legacy
    # {"type","factor"} shape and raises KeyError. 'default' == no scaling, so
    # forcing rope_scaling=None selects the correct unscaled RoPE branch.
    cfg = AutoConfig.from_pretrained(str(src_dir), trust_remote_code=True)
    cfg.rope_scaling = None
    tok = AutoTokenizer.from_pretrained(str(src_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(src_dir),
        config=cfg,
        trust_remote_code=True,
        dtype=torch_dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    ids_list: list[list[int]] = []
    logits_list: list[list[float]] = []
    with torch.no_grad():
        for p in prompts:
            ids = tok.encode(p, add_special_tokens=False)
            inp = torch.tensor([ids], device=device)
            # use_cache=False avoids the model's 4.x-era DynamicCache.from_legacy_cache
            # call, which transformers 5.x removed. A single forward needs no cache.
            out = model(inp, use_cache=False).logits[0, -1, :].to(torch.float32).cpu().tolist()
            ids_list.append(ids)
            logits_list.append(out)

    del model
    try:
        if device == "mps":
            torch.mps.empty_cache()  # type: ignore
    except Exception:
        pass
    return ids_list, logits_list


def _real_q_probe(src: Path) -> tuple[mx.array, str]:
    """Build a real-activation probe for the RoPE precision check.

    Returns q at layer 0 with shape ``(1, L, n_heads, head_dim)`` in fp32 --
    the tensor RoPE rotates -- computed as
    ``q_proj(input_layernorm(embed_tokens(prompt)))`` and reshaped to heads.
    Reuses the bisect shard readers (``_embed_rows`` + ``load_layer_torch``)
    so no full model is loaded. Layout matches the cos/sin broadcast in
    ``rope_precision_report`` (L, H, head_dim); RoPE is layout-agnostic as
    long as cos/sin and x agree.
    """
    import json

    from .bisect import _embed_rows, load_layer_torch

    from transformers import AutoTokenizer  # type: ignore

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
    ids = tok.encode("The capital of France is", add_special_tokens=False)
    L = len(ids)
    nh, hd = cfg["num_attention_heads"], cfg["head_dim"]
    eps = float(cfg["rms_norm_eps"])

    import torch  # type: ignore

    emb = _embed_rows(src, ids)                      # (L, H), fp32 numpy
    w = load_layer_torch(src, 0, "fp32")
    x = torch.from_numpy(emb).to(torch.float32)[None]   # (1, L, H)

    def rms(t, weight):
        v = t.pow(2).mean(-1, keepdim=True)
        return weight * (t * torch.rsqrt(v + eps))
    xn = rms(x, w["input_layernorm.weight"])          # input_layernorm, fp32
    q = torch.nn.functional.linear(xn, w["self_attn.q_proj.weight"])  # (1, L, nh*hd)
    q = q.view(1, L, nh, hd)                           # (1, L, nh, hd)
    q_mx = mx.array(q.contiguous().to(torch.float32).numpy())
    return q_mx, f"real q at layer 0 ({L} tokens, {nh} heads, head_dim {hd})"


def rope_precision_report(src: str | Path | None = None) -> dict[str, Any]:
    """Quantify the RoPE-precision floor in isolation -- isolated to cos/sin.

    What this measures and what it does NOT measure:

    * The HF reference downcasts cos/sin to bf16 (``return cos.to(dtype=x.dtype)``)
      before applying them; MLX keeps RoPE in fp32. With theta = 70 000 000 and
      head_dim 128, almost every frequency sits where ``cos ~= 1`` and bf16
      spacing near 1.0 is 2**-8 ~= 0.4%, so that downcast is a real floor.
    * The delta reported here is **only** the cos/sin-downcast effect. The
      previous version also rounded the probe x and did the multiply/add in bf16,
      which conflated three effects under one label. The reference keeps q in its
      compute dtype either way, so x-rounding is common-mode and must not appear
      in the delta. Here ``round_trig`` flips *only* the cos/sin bf16 downcast;
      x stays fp32 throughout both branches.

    Convention: precision measurements use real activations; only logic checks
    may use synthetic input. ``x`` below is the real ``q_proj(input_layernorm(
    embed_tokens))`` at layer 0, read straight from the shards -- no RNG in this
    path, so the metric is reproducible by construction (not by seeding).
    """
    head_dim = 128
    base = 70_000_000.0
    offset = 3
    half = head_dim // 2

    if src is not None:
        x, x_desc = _real_q_probe(Path(src))
        # probe layout is (1, L, H, head_dim)
        L, H = x.shape[1], x.shape[2]
    else:
        # No src provided (e.g. unit test): deterministic fallback so the function
        # stays callable. Never used by the parity report, which always passes src.
        L, H = 8, 2
        mx.random.seed(0)
        x = mx.random.normal((1, L, H, head_dim))
        x_desc = f"fallback probe (seeded, NOT real q): {H} heads, L={L}"

    def rotate_half(t):
        t1, t2 = t[..., :half], t[..., half:]
        return mx.concatenate([-t2, t1], axis=-1)

    def ref_rope(x_in, *, round_trig):
        # positions and frequencies are identical on both sides; only the
        # trig dtype differs. x_in is NEVER cast here -- that is the whole point.
        pos = mx.arange(offset, offset + L).astype(mx.float32)
        inv_freq = 1.0 / (base ** (mx.arange(0, half).astype(mx.float32) / half))
        freqs = mx.outer(pos, inv_freq)                 # (L, half)
        emb = mx.concatenate([freqs, freqs], axis=-1)   # (L, head_dim)
        cos, sin = mx.cos(emb), mx.sin(emb)
        if round_trig:                                  # emulate HF cos.to(bf16), nothing else
            cos = cos.astype(mx.bfloat16).astype(mx.float32)
            sin = sin.astype(mx.bfloat16).astype(mx.float32)
        cos = mx.broadcast_to(cos[:, None, :], (L, H, head_dim))[None]
        sin = mx.broadcast_to(sin[:, None, :], (L, H, head_dim))[None]
        return x_in * cos + rotate_half(x_in) * sin     # x stays fp32 throughout

    a = ref_rope(x, round_trig=False)   # ground truth: fp32 cos/sin
    b = ref_rope(x, round_trig=True)    # HF behavior: bf16 cos/sin, x unchanged

    def stats(u, v):
        u_l = u.reshape(-1).tolist()
        v_l = v.reshape(-1).tolist()
        dot = sum(p * q for p, q in zip(u_l, v_l))
        nu = math.sqrt(sum(p * p for p in u_l))
        nv = math.sqrt(sum(q * q for q in v_l))
        cos = dot / (nu * nv + 1e-12)
        m = max(abs(p - q) for p, q in zip(u_l, v_l))
        return {
            "cosine": round(cos, 6),
            "max_abs": round(m, 6),
            # absolute max_abs is scale-dependent; report it relative to ||q||
            # so the number is interpretable without knowing the probe magnitude.
            "max_abs_rel": round(m / (nu / math.sqrt(len(u_l)) + 1e-12), 6),
            "probe_rms": round(nu / math.sqrt(len(u_l)), 6),
        }

    s = stats(a, b)
    return {
        "probe": x_desc,
        "reference_fp32_cos_vs_bf16_cos": s,
        "interpretation": (
            f"Floor imposed by the HF reference's bf16 cos/sin downcast only "
            f"(x held in fp32 on both sides; the bf16 branch rounds cos/sin "
            f"exclusively). max_abs={s['max_abs']} ({s['max_abs_rel']} of probe "
            f"RMS {s['probe_rms']}). MLX computes RoPE in fp32 and avoids this, "
            "so the port is more accurate than the reference at this step; any "
            "strict logit comparison against the bf16 reference inherits roughly "
            "this floor from the reference side."
        ),
    }


def _softmax(x):
    m = max(x)
    ex = [math.exp(v - m) for v in x]
    s = sum(ex)
    return [e / s for e in ex]


def _kl(p, q):
    """KL(p || q) over the full vocab (both are probability lists)."""
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            total += pi * math.log(pi / qi if qi > 0 else 1e-12)
    return total


def run_parity(
    src_dir: str | Path,
    prompts: list[str] | None = None,
    output: str | Path | None = None,
    *,
    device: str = "cpu",
    dtype: str = "bf16",
    seed: int = 0,
) -> dict[str, Any]:
    """Run the fidelity comparison and (optionally) persist a JSON report."""
    prompts = prompts or DEFAULT_PROMPTS
    mx.random.seed(seed)

    ids_list, hf_logits = _gather_hf_logits(src_dir, prompts, device=device, dtype=dtype)
    mlx_model = load_mlx_bf16(src_dir)

    per_prompt = []
    top1_agree = 0
    cosines = []
    max_abs = []
    kls = []
    for ids, hf in zip(ids_list, hf_logits):
        mlx_logits = mlx_model(mx.array([ids]))[0, -1, :].astype(mx.float32)
        mlx_list = mlx_logits.tolist()
        hf_arr = list(hf)
        top1_hf = max(range(len(hf_arr)), key=lambda i: hf_arr[i])
        top1_mlx = max(range(len(mlx_list)), key=lambda i: mlx_list[i])
        agree = top1_hf == top1_mlx
        top1_agree += int(agree)
        # cosine
        dot = sum(a * b for a, b in zip(hf_arr, mlx_list))
        nh = math.sqrt(sum(a * a for a in hf_arr))
        nm = math.sqrt(sum(b * b for b in mlx_list))
        cos = dot / (nh * nm + 1e-12)
        cosines.append(cos)
        max_abs.append(max(abs(a - b) for a, b in zip(hf_arr, mlx_list)))
        kls.append(_kl(_softmax(hf_arr), _softmax(mlx_list)))
        per_prompt.append(
            {
                "prompt_len": len(ids),
                "top1_agree": agree,
                "cosine": round(cos, 6),
                "max_abs_logit": round(max_abs[-1], 4),
                "kl_p_to_q": round(kls[-1], 6),
                "top1_hf": top1_hf,
                "top1_mlx": top1_mlx,
            }
        )

    n = len(prompts)
    report = {
        "n_prompts": n,
        "device": device,
        "dtype": dtype,
        "seed": seed,
        "frameworks": {"reference": f"transformers ({device})", "port": "mlx"},
        "top1_agreement_rate": round(top1_agree / n, 4),
        "mean_cosine": round(sum(cosines) / n, 6),
        "min_cosine": round(min(cosines), 6),
        "mean_max_abs_logit": round(sum(max_abs) / n, 4),
        "mean_kl": round(sum(kls) / n, 6),
        "per_prompt": per_prompt,
        "rope_precision": rope_precision_report(src_dir),
        "interpretation": (
            "High top-1 agreement and logit cosine ≈ 1 indicate a faithful port. "
            "NOTE: the CPU run did NOT clear 0.99 — moving off MPS changed nothing "
            "(0.847 vs 0.844), so device numerics are ruled out as the cause. Do not "
            "read this number as settled; run `bisect --dtype fp32` before "
            "attributing the gap to precision (see rope_precision for "
            "the known bf16 floor). Bit-exact equality is not expected."
        ),
    }

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    del mlx_model
    return report
