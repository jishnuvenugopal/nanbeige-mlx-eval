"""mlx-nanbeige: an MLX port of the Nanbeige4.2-3B Looped Transformer.

The model definition lives in :mod:`mlx_nanbeige.model` and is the single
source of truth — it is also copied verbatim into each converted weight
directory as the ``model_file`` mlx-lm loads. Conversion (HF -> MLX quant) is
in :mod:`mlx_nanbeige.convert`; publishing helpers in
:mod:`mlx_nanbeige.upload` and :mod:`mlx_nanbeige.pull`.
"""

from __future__ import annotations

__version__ = "0.2.0"


def __getattr__(name: str):  # pragma: no cover - thin convenience export
    # Lazy re-export so ``from mlx_nanbeige import pull`` works without importing
    # huggingface_hub at package-import time.
    if name == "pull":
        from .pull import pull

        return pull
    raise AttributeError(f"module 'mlx_nanbeige' has no attribute {name!r}")
