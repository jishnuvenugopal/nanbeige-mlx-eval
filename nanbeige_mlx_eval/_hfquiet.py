"""Silence deprecation noise emitted by the checkpoint's vendored remote code.

Nanbeige4.2-3B ships `modeling_nanbeige.py` written against transformers 4.42.
Running it under transformers 5.x makes it print, on every forward:

    [transformers] `use_return_dict` is deprecated! Use `return_dict` instead!

That comes from ``self.config.use_return_dict`` inside the *vendored* modeling
file, not from this package, so it cannot be fixed here without patching
upstream code we deliberately keep pristine (`prepare_source` is non-mutating —
see convert.py). Passing ``return_dict=`` does not help: the property is read
before our argument is consulted.

Since the warning is upstream's and carries no information for our runs, filter
it. Call :func:`quiet_hf` immediately before loading a reference model. Anything
*other* than this specific deprecation still surfaces — the filter is narrow on
purpose, so a real warning is not swallowed with it.
"""

from __future__ import annotations

import warnings

_PATTERNS = (
    r"`use_return_dict` is deprecated",
    r"`?config\.use_return_dict`? is deprecated",
)

_applied = False


def quiet_hf() -> None:
    """Filter the vendored code's known-harmless deprecation warnings. Idempotent."""
    global _applied
    if _applied:
        return
    for pat in _PATTERNS:
        warnings.filterwarnings("ignore", message=pat, category=FutureWarning)
        warnings.filterwarnings("ignore", message=pat, category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=pat, category=UserWarning)

    # transformers routes some of these through its own logger rather than
    # `warnings`, which no filter will catch. Raise that logger's threshold just
    # past WARNING for the duration of the process; errors still print.
    try:
        from transformers.utils import logging as hf_logging  # type: ignore

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    _applied = True
