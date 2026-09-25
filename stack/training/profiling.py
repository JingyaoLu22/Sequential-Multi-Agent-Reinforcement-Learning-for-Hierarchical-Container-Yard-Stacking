"""
Lightweight wall-clock profiler for per-iteration phase timings.

Not a general-purpose profiler: every region this is used around is
coarse by design - one whole rollout, one whole PPO update phase, one
whole evaluation call, one checkpoint save - never a single minibatch.
This matters because a plain time.perf_counter() straddling
asynchronous CUDA kernels is misleading (the GPU work may still be
in flight when the timer stops), but calling torch.cuda.synchronize()
around every minibatch would itself add real overhead and partially
defeat the point of measuring the fast path. Restricting
synchronization to these coarse region boundaries avoids both
problems.

Profiling is opt-in (``--profile``): with it disabled, region() does
nothing at all, so normal training never synchronizes CUDA for timing.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Generator, Optional

import torch


class Profiler:
    """
    Accumulates named wall-clock durations across one training
    iteration, then hands them back once via pop_stats() so they can
    be merged into the same stats dict written to rollouts.csv.

    Parameters
    ----------
    enabled : bool
        When False, region() is a no-op context manager (no timing,
        no synchronization) so profiling can be disabled with
        negligible overhead.

    device : torch.device | None
        Only used to decide whether sync_cuda=True actually calls
        torch.cuda.synchronize() - never synchronizes on a CPU-only
        run even if a caller asks for it.
    """

    def __init__(
        self,
        enabled: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:

        self.enabled = bool(enabled)

        self._cuda_available = (
            device is not None
            and device.type == "cuda"
            and torch.cuda.is_available()
        )

        self._timings: Dict[str, float] = {}

    @contextmanager
    def region(
        self,
        name: str,
        sync_cuda: bool = False,
    ) -> Generator[None, None, None]:
        """
        Time one coarse region, accumulating into `name`.

        sync_cuda=True calls torch.cuda.synchronize() immediately
        before starting and immediately after stopping the clock, so
        the measured duration includes GPU work actually finishing,
        not just the CPU-side calls that launched it. Only pass this
        for a region that itself performs GPU work AND is coarse
        (a full rollout, a full PPO update phase, a full evaluation) -
        never inside a minibatch loop.
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

            elapsed = time.perf_counter() - start

            self._timings[name] = (
                self._timings.get(name, 0.0)
                + elapsed
            )

    def pop_stats(self) -> Dict[str, float]:
        """
        Return accumulated timings since the last pop_stats() call
        (or construction), and reset the accumulator.
        """

        stats = dict(self._timings)
        self._timings = {}
        return stats
