# Copyright © 2026 Jishnu Venugopal
#
# MLX port of the Nanbeige *Looped Transformer* (Nanbeige4.2-3B).
#
# This file is the single source of truth for the port. It is dual-purpose:
#   1. Importable as ``mlx_nanbeige.model`` (the package this file ships in).
#   2. A self-contained ``model_file`` that mlx-lm can import directly from a
#      converted weight directory (see ``config.json`` -> ``"model_file"``), so
#      anyone can ``mlx_lm.load`` a converted Nanbeige repo without a registry
#      entry. The conversion step copies this file verbatim into each quant dir.
#
# Dependency surface of the shipped ``model_file``:
#   The trivial helpers (SwiGLU, the unscaled RoPE branch, BaseModelArgs
#   field-filtering) are inlined below so the shipped file has no dependency on
#   mlx-lm's private modules for behavior it can express directly. The helpers
#   that carry real behavior (``scaled_dot_product_attention`` for quantized-KV
#   dispatch, ``KVCache``, ``create_attention_mask``) are imported from
#   ``mlx_lm.models.*`` behind a single guarded import that raises an actionable
#   error. This file is therefore **pinned to mlx-lm >= 0.31 internals** (tested
#   0.31.3); an upstream mlx-lm refactor of those modules would require a new
#   model_file. That's an honest trade for keeping quantized-KV support.
#
# Only absolute ``mlx`` / ``mlx_lm`` imports are used so the file works
# standalone when dropped into a weight repo.
#
# Architecture for the Nanbeige4.2-3B checkpoint (see its ``config.json``):
#   - Standard Llama-style decoder blocks: RMSNorm -> GQA attention -> residual,
#     RMSNorm -> SwiGLU MLP -> residual.
#   - ``num_loops = 2``: the 22 physical layers are executed twice per forward
#     pass (weight-shared recurrence). The final RMSNorm is applied at *each*
#     loop boundary (``skip_loop_final_norm = False``), i.e. after the 22nd and
#     after the 44th effective layer.
#   - GQA: 48 query heads, 8 key/value heads.
#   - ``head_dim = 128`` while ``hidden_size / num_heads = 3072 / 48 = 64``.
#     The projection widths therefore use the explicit head_dim, giving
#     non-square q/o projections (q: 3072 -> 6144, o: 6144 -> 3072).
#   - RoPE is the rotate-half convention (MLX ``rope_traditional=False``) with an
#     unusually large base ``rope_theta = 70_000_000`` for the 256K context.
#   - The richer Nanbeige config class also supports n-gram embeddings, hyper /
#     mini-hyper connection and depth attention, but this checkpoint leaves all
#     of those disabled, so the port implements the plain looped variant.

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

# --- mlx-lm internals this model_file depends on (pinned to >= 0.31) ---------
# These carry real behavior (quantized-KV dispatch in particular) that we want
# to keep, so they are guarded rather than inlined. An actionable error beats a
# raw ModuleNotFoundError in a stranger's traceback years from now.
try:
    from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
    from mlx_lm.models.cache import KVCache
except ImportError as exc:  # pragma: no cover - exercised only on bad installs
    raise ImportError(
        "mlx_nanbeige requires mlx-lm >= 0.31 (tested 0.31.3) and mlx >= 0.32. "
        "The internal helpers this model_file uses "
        "(mlx_lm.models.base.{scaled_dot_product_attention,create_attention_mask}, "
        "mlx_lm.models.cache.KVCache) moved or were removed in your version. "
        "Upgrade: pip install -U 'mlx-lm>=0.31' 'mlx>=0.32'."
    ) from exc


def _swiglu(gate: mx.array, x: mx.array) -> mx.array:
    """SwiGLU activation: ``silu(gate) * x`` (reference: ``act_fn(gate) * up``).

    Inlined from ``mlx_lm.models.activations.swiglu``; the only difference is
    the upstream copy is wrapped in ``mx.compile`` (a speed optimization, not a
    behavior change), which we deliberately do not bake into a shipped
    ``model_file`` — mlx-lm applies its own compilation at generate time.
    """
    return nn.silu(gate) * x


def _make_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: Optional[Dict[str, Union[float, str]]],
    max_position_embeddings: Optional[int],
) -> nn.RoPE:
    """Construct the RoPE module.

    The unscaled branch (``scaling_config is None``) is inlined as
    ``nn.RoPE(dims, traditional, base)`` so the shipped file does not depend on
    ``mlx_lm.models.rope_utils`` for the common case. Scaled RoPE variants
    (Llama-3 / dynamic / Yarn / ...) are deferred to ``initialize_rope`` behind a
    guarded import — this checkpoint uses no scaling, but the branch is kept so
    the file stays correct if a future Nanbeige checkpoint enables one.
    """
    if scaling_config is None:
        return nn.RoPE(dims, traditional=traditional, base=base)
    try:
        from mlx_lm.models.rope_utils import initialize_rope
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scaled RoPE requires mlx-lm >= 0.31 (mlx_lm.models.rope_utils)."
        ) from exc
    return initialize_rope(
        dims,
        base=base,
        traditional=traditional,
        scaling_config=scaling_config,
        max_position_embeddings=max_position_embeddings,
    )


@dataclass
class ModelArgs:
    """Configuration for the Nanbeige looped transformer.

    Field-filtering on construction (``from_dict``) is inlined from
    ``mlx_lm.models.base.BaseModelArgs`` so this dataclass has no mlx-lm base
    dependency in the shipped file.
    """

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float = 70_000_000.0
    num_loops: int = 1
    max_position_embeddings: int = 262144
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelArgs":
        """Build from a config dict, ignoring keys this dataclass doesn't define."""
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Attention(nn.Module):
    """Grouped-query attention with an explicit (non-default) head dimension."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        # NOTE: head_dim is taken from the config (128), not hidden_size // n_heads.
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        # bias follows config.attention_bias (harmless: this checkpoint is False).
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = _make_rope(
            head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    """A standard pre-norm Llama-style decoder layer."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class NanbeigeModel(nn.Module):
    """The looped transformer trunk.

    ``num_loops`` passes are made over the same ``num_hidden_layers`` physical
    blocks (weight sharing). The final RMSNorm is applied at the end of every
    pass, matching ``skip_loop_final_norm = False`` in the reference checkpoint.

    Because each (loop, layer) pair attends over a distinct key/value stream,
    the cache needs ``num_loops * num_hidden_layers`` virtual slots even though
    only ``num_hidden_layers`` blocks of weights exist.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_hidden_layers = args.num_hidden_layers
        self.num_loops = args.num_loops
        self.vocab_size = args.vocab_size

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * (self.num_loops * self.num_hidden_layers)

        # All virtual cache slots stay in lockstep (same offset) within a single
        # call, so a single mask derived from the first slot is valid for every
        # layer in both loops.
        mask = create_attention_mask(h, cache[0])

        for loop in range(self.num_loops):
            for i, layer in enumerate(self.layers):
                # Virtual index: each (loop, layer) pair owns its own KV slot so
                # the two passes over the shared weights do not collide.
                h = layer(h, mask, cache[loop * self.num_hidden_layers + i])
            h = self.norm(h)

        return h


class Model(nn.Module):
    """Top-level causal LM wrapper, conforming to the mlx-lm model contract."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = NanbeigeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def make_cache(self):
        """Allocate the loop-aware KV cache (``num_loops * num_hidden_layers`` slots).

        ``mlx_lm``'s :func:`make_prompt_cache` defers to this method when present,
        which is what lets a model with weight-shared looped layers expose the
        larger effective depth the cache requires.

        Note on the ``--max-kv-size`` knob: ``make_prompt_cache`` ignores
        ``max_kv_size`` whenever a model supplies ``make_cache``, so the
        rotating-cache / max-length knob is **inert** for this model.
        ``--kv-bits`` still works (``KVCache.to_quantized``). The practical
        ceiling is real: at full context the looped design needs
        ``num_loops * num_hidden_layers`` KV slots, i.e. for this checkpoint
        44 x 8 KV-heads x 128 dim x 2 (K+V) x 262144 positions x 2 bytes
        ~= 47 GB — unreachable on a 16 GB machine. This is a genuine,
        quantifiable cost of the looped design, not a bug.
        """
        n = self.args.num_loops * self.args.num_hidden_layers
        return [KVCache() for _ in range(n)]

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def sanitize(self, weights):
        """Weight keys map 1:1 to the module tree for this checkpoint.

        Drop any non-persistent buffers a checkpoint might carry (e.g. a cached
        rotary ``inv_freq``) so a strict load succeeds.
        """
        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
