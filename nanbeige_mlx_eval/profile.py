"""Lightweight latency / memory profiling for MLX generation.

On Apple Silicon the GPU uses unified memory. ``mx.get_peak_memory`` is the
meaningful per-run allocator high-water mark for Metal buffers; the process
``ru_maxrss`` is a monotonic process-wide high-water mark that carries no
per-case information, kept here only as a secondary ``process_rss_mb`` field.
``mx.reset_peak_memory`` must be called between cases for the per-case number
to mean anything.
"""

from __future__ import annotations

import math
import resource
import time
from dataclasses import dataclass


@dataclass
class GenProfile:
    prompt_tokens: int
    generated_tokens: int
    total_seconds: float
    ttft_seconds: float
    peak_rss_mb: float

    @property
    def tokens_per_second(self) -> float:
        return (self.generated_tokens / self.total_seconds) if self.total_seconds else 0.0


def peak_rss_mb() -> float:
    """Process resident set size (macOS: ``ru_maxrss`` is bytes). Monotonic.

    Secondary metric — identical for every case in a run, so it carries no
    per-case information. Prefer :func:`mlx_peak_mb` for the allocator number.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def mlx_peak_mb() -> float:
    """Peak Metal-buffer allocation since the last reset, in MB.

    Requires ``mlx.core.reset_peak_memory`` to have been called before the
    measured region. Returns 0.0 on mlx builds without the allocator counters.
    """
    import mlx.core as mx

    try:
        bytes_ = mx.get_peak_memory()
    except Exception:
        return 0.0
    if bytes_ is None:
        return 0.0
    return float(bytes_) / (1024.0 * 1024.0)


def reset_peak_memory() -> None:
    """Reset the mlx allocator's peak-memory counter (no-op if unsupported)."""
    import mlx.core as mx

    try:
        mx.reset_peak_memory()
    except Exception:
        pass


def now() -> float:
    return time.perf_counter()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial pass rate.

    ``k`` passes out of ``n``; returns the (low, high) 95% CI (for z=1.96).
    Use this instead of a bare ``k/n`` so that 8/8 is honestly reported as
    ``1.00 [0.67, 1.00]`` rather than implying certainty.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - spread, center + spread)
