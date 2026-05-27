import numpy as np
import pytest

from sdbench.equivalence import compare_to_reference
from sdbench.timing import run_timed_steps


class IncrementAdapter:
    def __init__(self):
        self.calls = 0

    def step(self, latent, timestep, text_embedding):
        self.calls += 1
        return latent + self.calls


def test_timing_discards_cold_run_and_reports_median_iqr():
    adapter = IncrementAdapter()
    latent = np.zeros((1, 1), dtype=np.float32)
    text_embedding = np.zeros((1, 1), dtype=np.float32)

    result = run_timed_steps(
        adapter=adapter,
        latent=latent,
        timestep=1,
        text_embedding=text_embedding,
        warmup=1,
        iterations=10,
        clock_values=[
            100.0,
            101.0,
            200.0,
            202.0,
            300.0,
            303.0,
            400.0,
            404.0,
            500.0,
            505.0,
            600.0,
            606.0,
            700.0,
            707.0,
            800.0,
            808.0,
            900.0,
            909.0,
            1000.0,
            1010.0,
        ],
    )

    assert adapter.calls == 11
    assert result.latency_ms_median == pytest.approx(5.5)
    assert result.latency_ms_iqr == pytest.approx(4.5)
    assert len(result.iteration_windows) == 10
    np.testing.assert_array_equal(result.last_output, np.array([[11.0]], dtype=np.float32))


def test_equivalence_computes_mse_cosine_and_flags_threshold_failures():
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    output = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    result = compare_to_reference(output, reference, mse_max=1.0e-3, cosine_min=0.999)

    assert result.mse > 0.1
    assert result.cosine == 0.0
    assert result.passed is False
