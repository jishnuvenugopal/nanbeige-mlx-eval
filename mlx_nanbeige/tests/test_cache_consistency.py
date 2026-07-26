"""Prefill-vs-decode self-consistency for the looped KV cache.

This is the test the 44-slot scheme actually demands, and it needs no reference
model — it runs in milliseconds on a tiny random model. Two virtual cache slots
per physical layer (one per loop) must stay in lockstep; if the loop/slot index
were ever swapped, the incremental decode would diverge from the one-shot
prefill and the offsets would drift.

Addresses A2 in CODE_REVIEW.md.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_nanbeige.model import Model, ModelArgs


def _tiny_model(num_loops: int = 2, num_layers: int = 2) -> Model:
    args = ModelArgs(
        model_type="nanbeige",
        hidden_size=32,
        num_hidden_layers=num_layers,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-5,
        vocab_size=64,
        num_loops=num_loops,
    )
    m = Model(args)
    mx.eval(m.parameters())
    return m


def test_make_cache_slot_count():
    """The cache exposes num_loops * num_hidden_layers virtual slots."""
    m = _tiny_model(num_loops=2, num_layers=2)
    cache = m.make_cache()
    assert len(cache) == 4
    m2 = _tiny_model(num_loops=3, num_layers=5)
    assert len(m2.make_cache()) == 15


def test_prefill_matches_incremental_decode():
    """Two virtual cache slots per layer must stay in lockstep.

    Generates N-1 tokens incrementally, then steps the Nth; the last-token
    logits must match a one-shot prefill of the full sequence. atol is generous
    (2e-2) because MLX is non-deterministic across batch shapes at the margin;
    a swapped loop index diverges far more than that.
    """
    m = _tiny_model(num_loops=2, num_layers=2)
    ids = mx.array([[5, 9, 3, 7, 2, 8]])

    full = m(ids)[0, -1].astype(mx.float32)

    cache = m.make_cache()
    for i in range(ids.shape[1] - 1):
        m(ids[:, i : i + 1], cache=cache)
    step = m(ids[:, -1:], cache=cache)[0, -1].astype(mx.float32)

    assert mx.allclose(full, step, atol=2e-2).item(), (
        "prefill vs incremental decode diverged: "
        f"max abs diff {float(mx.max(mx.abs(full - step)))}"
    )


def test_cache_offsets_in_lockstep():
    """After a forward, every virtual slot must have advanced to the same offset."""
    m = _tiny_model(num_loops=2, num_layers=2)
    ids = mx.array([[5, 9, 3, 7, 2, 8]])
    cache = m.make_cache()
    m(ids, cache=cache)
    offsets = [c.offset for c in cache]
    assert all(o == ids.shape[1] for o in offsets), (
        f"cache slots drifted out of lockstep: {offsets}"
    )


def test_cache_offset_matches_across_loops():
    """Slots for loop 0 and loop 1 must share offsets (the per-loop streams co-evolve).

    With num_loops=2 and num_hidden_layers=2, slots 0..1 are loop-0 layers and
    slots 2..3 are loop-1 layers; both halves must be identical.
    """
    n_layers = 3
    m = _tiny_model(num_loops=2, num_layers=n_layers)
    ids = mx.array([[1, 2, 3, 4]])
    cache = m.make_cache()
    m(ids, cache=cache)
    offsets = [c.offset for c in cache]
    loop0 = offsets[:n_layers]
    loop1 = offsets[n_layers:]
    assert loop0 == loop1 == [ids.shape[1]] * n_layers, (
        f"loop-0 / loop-1 offsets disagree: {loop0} vs {loop1}"
    )
