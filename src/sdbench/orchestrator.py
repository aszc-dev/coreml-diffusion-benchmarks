from pathlib import Path

import numpy as np

from sdbench.adapter import RealizedConfig
from sdbench.config import BenchmarkConfig, CellConfig
from sdbench.equivalence import compare_to_reference
from sdbench.inputs import SharedInput
from sdbench.results import BenchmarkRecord
from sdbench.sizing import ModelSize
from sdbench.timing import run_timed_steps


def run_matrix(
    cfg: BenchmarkConfig,
    shared_input: SharedInput,
    adapters: dict[str, object],
    run_id: str,
    results_dir: str | Path,
) -> list[BenchmarkRecord]:
    records = []
    reference = shared_input.latent
    for cell in cfg.cells:
        records.append(_run_cell(cfg, cell, shared_input, adapters, run_id, reference))
    return records


def _run_cell(
    cfg: BenchmarkConfig,
    cell: CellConfig,
    shared_input: SharedInput,
    adapters: dict[str, object],
    run_id: str,
    reference: np.ndarray,
) -> BenchmarkRecord:
    adapter = adapters[cell.backend]
    realized: RealizedConfig | None = None
    try:
        realized = adapter.prepare(cell)
        timing = run_timed_steps(
            adapter=adapter,
            latent=shared_input.latent,
            timestep=shared_input.timestep,
            text_embedding=shared_input.text_embedding,
            warmup=cfg.warmup,
            iterations=cfg.iterations,
        )
        reference_output = _reference_output(adapter, shared_input, reference)
        equivalence = compare_to_reference(
            timing.last_output,
            reference_output,
            mse_max=cfg.equivalence.mse_max,
            cosine_min=cfg.equivalence.cosine_min,
        )
        # Power is filled in post-hoc from the powermetrics capture (see cli.py `power`
        # command); the sampler runs concurrently, so samples don't exist yet here.
        model_size = _model_size(adapter)
        return BenchmarkRecord(
            run_id=run_id,
            cell_id=cell.id,
            backend=cell.backend,
            requested_compute_unit=cell.compute_unit,
            realized_compute_unit=realized.compute_unit,
            attention=realized.attention,
            precision=realized.precision,
            resolution=cell.resolution,
            status="ok",
            latency_ms_median=timing.latency_ms_median,
            latency_ms_iqr=timing.latency_ms_iqr,
            gpu_power_w=None,
            ane_power_w=None,
            energy_per_unet_step_j=None,
            estimated_energy_per_50_step_image_j=None,
            mse=equivalence.mse,
            cosine=equivalence.cosine,
            numerically_divergent=not equivalence.passed,
            on_disk_size_bytes=model_size.on_disk_size_bytes if model_size else None,
            weight_only_size_bytes=model_size.weight_only_size_bytes if model_size else None,
            effective_bits_per_parameter=model_size.effective_bits_per_parameter if model_size else None,
            compute_precision=model_size.compute_precision if model_size else realized.precision,
            graph_capture_s=None,
            convert_s=None,
            first_load_compile_s=None,
            failure_reason=None,
            active_window_start_s=timing.active_wall_start_s,
            active_window_end_s=timing.active_wall_end_s,
        )
    except Exception as exc:
        return _failed_record(run_id, cell, realized, str(exc))
    finally:
        adapter.teardown()


def _failed_record(
    run_id: str,
    cell: CellConfig,
    realized: RealizedConfig | None,
    reason: str,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        run_id=run_id,
        cell_id=cell.id,
        backend=cell.backend,
        requested_compute_unit=cell.compute_unit,
        realized_compute_unit=realized.compute_unit if realized else None,
        attention=realized.attention if realized else cell.attention,
        precision=realized.precision if realized else cell.precision,
        resolution=cell.resolution,
        status="failed",
        latency_ms_median=None,
        latency_ms_iqr=None,
        gpu_power_w=None,
        ane_power_w=None,
        energy_per_unet_step_j=None,
        estimated_energy_per_50_step_image_j=None,
        mse=None,
        cosine=None,
        numerically_divergent=None,
        on_disk_size_bytes=None,
        weight_only_size_bytes=None,
        effective_bits_per_parameter=None,
        compute_precision=realized.precision if realized else None,
        graph_capture_s=None,
        convert_s=None,
        first_load_compile_s=None,
        failure_reason=reason,
    )


def _reference_output(adapter, shared_input: SharedInput, fallback: np.ndarray) -> np.ndarray:
    if hasattr(adapter, "reference_step"):
        return adapter.reference_step(
            shared_input.latent,
            shared_input.timestep,
            shared_input.text_embedding,
        )
    return fallback


def _model_size(adapter) -> ModelSize | None:
    if hasattr(adapter, "model_size"):
        return adapter.model_size()
    return None
