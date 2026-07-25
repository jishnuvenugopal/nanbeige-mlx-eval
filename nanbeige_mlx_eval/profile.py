"""Lightweight latency / memory profiling for MLX generation.

On Apple Silicon the GPU uses unified memory, so the process resident size
captures the model footprint. ``resource.getrusage`` reports ``ru_maxrss`` in
*bytes* on macOS (unlike Linux, where it is KiB).
"""

from __future__ import annotations

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
    """Peak resident set size of the process in megabytes (macOS byte units)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def now() -> float:
    return time.perf_counter()
