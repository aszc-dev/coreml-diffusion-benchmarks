import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class IterationWindow:
    start_s: float
    end_s: float
    latency_ms: float


@dataclass(frozen=True)
class TimingResult:
    latency_ms_median: float
    latency_ms_iqr: float
    latency_ms_min: float
    latency_ms_max: float
    iteration_windows: list[IterationWindow]
    last_output: np.ndarray
    # Wall-clock (epoch) bounds of the timed window, for post-hoc power alignment.
    # Latency stays monotonic-derived; only alignment needs a cross-process clock (R6.3).
    active_wall_start_s: float | None = None
    active_wall_end_s: float | None = None


def run_timed_steps(
    adapter,
    latent: np.ndarray,
    timestep: int,
    text_embedding: np.ndarray,
    warmup: int,
    iterations: int,
    clock_values: Iterable[float] | None = None,
    on_warmup: Callable[[int, int, float], None] | None = None,
    on_timed: Callable[[int, int, float], None] | None = None,
) -> TimingResult:
    if iterations < 10:
        raise ValueError("Benchmark must run at least 10 timed iterations")

    clock = _Clock(clock_values)
    output = latent
    for w in range(warmup):
        # Only touch the clock when reporting, so the injected-clock budget used by
        # tests (sized for the timed iterations) is untouched on the default path.
        if on_warmup is not None:
            start = clock.now()
            output = np.asarray(adapter.step(latent, timestep, text_embedding))
            on_warmup(w + 1, warmup, (clock.now() - start) * 1000.0)
        else:
            output = np.asarray(adapter.step(latent, timestep, text_embedding))

    windows: list[IterationWindow] = []
    wall_start: float | None = None
    wall_end: float | None = None
    for i in range(iterations):
        if wall_start is None:
            wall_start = time.time()
        start = clock.now()
        output = np.asarray(adapter.step(latent, timestep, text_embedding))
        end = clock.now()
        wall_end = time.time()
        windows.append(IterationWindow(start_s=start, end_s=end, latency_ms=(end - start) * 1000.0))
        if on_timed is not None:
            on_timed(i + 1, iterations, windows[-1].latency_ms)

    latencies = np.array([window.latency_ms for window in windows], dtype=np.float64)
    q1, q3 = np.percentile(latencies, [25, 75])
    return TimingResult(
        latency_ms_median=float(np.median(latencies)),
        latency_ms_iqr=float(q3 - q1),
        latency_ms_min=float(np.min(latencies)),
        latency_ms_max=float(np.max(latencies)),
        iteration_windows=windows,
        last_output=output,
        active_wall_start_s=wall_start,
        active_wall_end_s=wall_end,
    )


class _Clock:
    def __init__(self, values: Iterable[float] | None):
        self.values = iter(values) if values is not None else None

    def now(self) -> float:
        if self.values is None:
            return time.monotonic()
        return float(next(self.values)) / 1000.0
