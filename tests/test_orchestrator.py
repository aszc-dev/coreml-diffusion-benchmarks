import numpy as np

from sdbench.adapter import RealizedConfig
from sdbench.config import BenchmarkConfig, CellConfig, EquivalenceConfig, PowerConfig, ThermalConfig
from sdbench.inputs import SharedInput
from sdbench.orchestrator import run_matrix


class PassingAdapter:
    name = "passing"

    def prepare(self, cfg):
        return RealizedConfig(
            compute_unit=cfg.compute_unit,
            attention=cfg.attention,
            precision=cfg.precision,
            artifact_paths=[],
        )

    def step(self, latent, timestep, text_embedding):
        return latent

    def teardown(self):
        pass


class FailingAdapter:
    name = "failing"

    def prepare(self, cfg):
        raise RuntimeError("backend unavailable")

    def step(self, latent, timestep, text_embedding):
        return latent

    def teardown(self):
        pass


def test_orchestrator_records_failure_and_continues(tmp_path):
    cfg = BenchmarkConfig(
        checkpoint=tmp_path / "sd15.safetensors",
        seed=0,
        iterations=10,
        warmup=1,
        thermal=ThermalConfig(throttle_policy="flag"),
        equivalence=EquivalenceConfig(mse_max=1.0e-3, cosine_min=0.999),
        power=PowerConfig(interval_ms=100, baseline_seconds=2),
        cells=[
            CellConfig(id="fail", backend="failing", compute_unit="GPU", attention="NATIVE", precision="fp16", resolution=512),
            CellConfig(id="pass", backend="passing", compute_unit="GPU", attention="NATIVE", precision="fp16", resolution=512),
        ],
    )
    shared = SharedInput(
        latent=np.zeros((1, 1), dtype=np.float32),
        timestep=1,
        text_embedding=np.zeros((1, 1), dtype=np.float32),
    )

    records = run_matrix(
        cfg=cfg,
        shared_input=shared,
        adapters={"failing": FailingAdapter(), "passing": PassingAdapter()},
        run_id="test-run",
        results_dir=tmp_path,
    )

    assert [record.cell_id for record in records] == ["fail", "pass"]
    assert records[0].status == "failed"
    assert "backend unavailable" in records[0].failure_reason
    assert records[1].status == "ok"
