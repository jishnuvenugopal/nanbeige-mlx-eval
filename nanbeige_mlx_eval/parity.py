"""Half A — fidelity gate: MLX port vs the HuggingFace reference.

A from-scratch MLX port of a custom architecture cannot be trusted on coherent
generation alone. This module runs the same prompts through:

* the official Nanbeige checkpoint under ``transformers`` (``trust_remote_code``,
  bfloat16, on MPS), and
* this project's MLX port (bfloat16, loaded in memory from the source weights),

and compares next-token logits. The HF model is loaded, measured and freed
*before* the MLX model is loaded, so the two never co-reside (16 GB machine).

Because the two frameworks differ in matmul ordering and RMSNorm upcasting,
bit-exact equality is not expected; the bar is high *agreement* (top-1 identity
rate, logit cosine, KL). Numbers are reported honestly, not thresholded into a
false "identical" claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx

from .models.nanbeige import Model, ModelArgs


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


def _gather_hf_logits(src_dir: str | Path, prompts: list[str]) -> tuple[list, list]:
    """Load HF model, return (input_ids_list, last_token_logits_list) in fp32."""
    import torch  # type: ignore
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore

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
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("mps")
    model.eval()

    ids_list: list[list[int]] = []
    logits_list: list[list[float]] = []
    with torch.no_grad():
        for p in prompts:
            ids = tok.encode(p, add_special_tokens=False)
            inp = torch.tensor([ids], device="mps")
            # use_cache=False avoids the model's 4.x-era DynamicCache.from_legacy_cache
            # call, which transformers 5.x removed. A single forward needs no cache.
            out = model(inp, use_cache=False).logits[0, -1, :].to(torch.float32).cpu().tolist()
            ids_list.append(ids)
            logits_list.append(out)

    del model
    try:
        torch.mps.empty_cache()  # type: ignore
    except Exception:
        pass
    return ids_list, logits_list


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
) -> dict[str, Any]:
    """Run the fidelity comparison and (optionally) persist a JSON report."""
    prompts = prompts or DEFAULT_PROMPTS

    ids_list, hf_logits = _gather_hf_logits(src_dir, prompts)
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
        "dtype": "bfloat16",
        "frameworks": {"reference": "transformers (MPS)", "port": "mlx"},
        "top1_agreement_rate": round(top1_agree / n, 4),
        "mean_cosine": round(sum(cosines) / n, 6),
        "min_cosine": round(min(cosines), 6),
        "mean_max_abs_logit": round(sum(max_abs) / n, 4),
        "mean_kl": round(sum(kls) / n, 6),
        "per_prompt": per_prompt,
        "interpretation": (
            "High top-1 agreement and logit cosine ≈ 1 indicate a faithful port. "
            "Bit-exact equality is not expected across frameworks at bfloat16."
        ),
    }

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    del mlx_model
    return report
