"""
Opt-in (``--profile``) wall-clock timings of coarse training phases:
a whole rollout, PPO update phase, evaluation or checkpoint, never a
single minibatch or environment step. Disabled, region() does nothing,
so normal training never synchronizes CUDA for timing.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Generator, Optional

import torch


class Profiler:
    """Accumulates named durations until pop_stats() hands them back."""

    def __init__(self, enabled: bool = False, device: Optional[torch.device] = None) -> None:
        self.enabled = bool(enabled)
        # sync_cuda only synchronizes when training actually runs on a GPU.
        self._cuda_available = device is not None and device.type == "cuda" and torch.cuda.is_available()
        self._timings: Dict[str, float] = {}

    @contextmanager
    def region(self, name: str, sync_cuda: bool = False) -> Generator[None, None, None]:
        """Add the duration of the with-block to ``name``.

        sync_cuda=True synchronizes CUDA at both ends so queued GPU work
        is counted; use it only around coarse regions that launch GPU work.
        """

        if not self.enabled:
            yield
            return

        if sync_cuda and self._cuda_available:
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync_cuda and self._cuda_available:
                torch.cuda.synchronize()
            self._timings[name] = self._timings.get(name, 0.0) + time.perf_counter() - start

    def pop_stats(self) -> Dict[str, float]:
        """Timings accumulated since the last call; resets them."""

        stats, self._timings = self._timings, {}
        return stats
