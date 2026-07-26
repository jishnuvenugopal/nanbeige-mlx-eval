"""Download a published Nanbeige4.2-3B MLX quant by short name.

Wraps :func:`huggingface_hub.snapshot_download` with the repo ids baked in, so
the user experience is a one-liner with no conversion step::

    from mlx_nanbeige import pull
    import mlx_lm
    model, tok = mlx_lm.load(pull("4bit"))

Weights arrive on first use, cached in ``~/.cache/huggingface``, resumable and
revision-pinnable — indistinguishable from bundling, and exactly how
``mlx-community`` ships every model.
"""

from __future__ import annotations

QUANTS: dict[str, str] = {
    "4bit": "jishnuvenugopal/Nanbeige4.2-3B-mlx-4bit",
    "6bit": "jishnuvenugopal/Nanbeige4.2-3B-mlx-6bit",
    "8bit": "jishnuvenugopal/Nanbeige4.2-3B-mlx-8bit",
}


def pull(quant: str = "4bit", revision: str | None = None) -> str:
    """Download a published quant; returns a local path for ``mlx_lm.load``.

    Parameters
    ----------
    quant:
        One of ``"4bit"``, ``"6bit"``, ``"8bit"`` (see :data:`QUANTS`).
    revision:
        Optional pinned HF revision (commit SHA or tag). Pinning is recommended
        for reproducibility — see the eval README, which records exact SHAs.
    """
    if quant not in QUANTS:
        raise ValueError(
            f"unknown quant {quant!r}; choose one of {sorted(QUANTS)}"
        )
    from huggingface_hub import snapshot_download  # type: ignore

    return snapshot_download(repo_id=QUANTS[quant], revision=revision)
