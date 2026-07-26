# mlx-nanbeige

An MLX port of the **Nanbeige4.2-3B *Looped Transformer*** for Apple Silicon,
plus the HF→MLX conversion and publishing helpers. The model definition in
[`mlx_nanbeige/model.py`](mlx_nanbeige/model.py) is the **single source of truth**
for the port — it is also copied verbatim into each converted weight directory
as the `model_file` mlx-lm loads.

This is an independent project; it is not affiliated with or endorsed by the
Nanbeige team.

## What's here

| module | purpose |
|---|---|
| `mlx_nanbeige/model.py` | the port — also the shipped `model_file` |
| `mlx_nanbeige/convert.py` | HF → MLX quant (non-mutating staging, tokenizer verify) |
| `mlx_nanbeige/upload.py` | model-card + LICENSE + NOTICE + (opt-in) HF upload |
| `mlx_nanbeige/pull.py` | `pull("4bit")` → local path for `mlx_lm.load` |

## Load a published quant (one line)

```python
from mlx_nanbeige import pull
import mlx_lm
model, tok = mlx_lm.load(pull("4bit"))
```

## Convert from the BF16 checkpoint yourself

```bash
mlx-nanbeige-convert --src /path/to/Nanbeige4.2-3B --out ./nanbeige-mlx-4bit --bits 4
```

The source directory is never mutated; the tokenizer round-trip is asserted.

## The 44-slot KV cache (a real cost of the looped design)

The looped architecture needs `num_loops * num_hidden_layers = 44` KV slots.
At full context that is 44 × 8 KV-heads × 128 dim × 2 (K+V) × 262 144 positions
× 2 bytes ≈ **47 GB** — unreachable on a 16 GB machine. Because this model
supplies `make_cache`, mlx-lm's `--max-kv-size` knob is inert; use `--kv-bits`
to reduce KV precision instead. This is a quantifiable cost of the looped
design, not a bug — see `make_cache()`'s docstring.

## Dependencies

Pinned to `mlx>=0.32` and `mlx-lm>=0.31` (the shipped `model_file` uses
`mlx_lm.models.base` / `.cache` internals; see `model.py`).

## License

MIT for the code in this package. The Nanbeige model weights are governed by
the upstream Apache-2.0 license; convert and redistribute them per that license.
