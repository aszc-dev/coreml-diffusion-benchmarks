from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sdbench import telemetry
from sdbench.adapter import RealizedConfig
from sdbench.config import BenchmarkConfig, CellConfig
from sdbench.equivalence import compare_to_reference
from sdbench.inputs import SharedInput
from sdbench.progress import NullReporter, RunReporter
from sdbench.results import BenchmarkRecord
from sdbench.sizing import ModelSize
from sdbench.telemetry import TELEMETRY_SCHEMA_VERSION
from sdbench.timing import run_timed_steps


@dataclass(frozen=True)
class TelemetryContext:
    """Per-run static telemetry stamped on every record (R11.6, R11.11, R11.13).

    Computed once at run start and passed to every cell so a single JSONL row is
    self-describing about the host and the input it ran under, without re-probing
    sysctl/git/uv on every cell."""

    host_id_hash: str | None = None
    env_vars_digest: str | None = None
    power_sampler_interval_ms: int | None = None
    latent_input_sha256: str | None = None
    text_embedding_input_sha256: str | None = None
    schema_version: int = TELEMETRY_SCHEMA_VERSION


def _empty_context() -> TelemetryContext:
    return TelemetryContext()


def run_matrix(
    cfg: BenchmarkConfig,
    shared_input: SharedInput,
    adapters: dict[str, object],
    run_id: str,
    results_dir: str | Path,
    reporter: RunReporter | None = None,
    telemetry_ctx: TelemetryContext | None = None,
) -> list[BenchmarkRecord]:
    reporter = reporter or NullReporter()
    ctx = telemetry_ctx or _empty_context()
    records = []
    reference = shared_input.latent
    total = len(cfg.cells)
    reporter.run_start(total)
    for index, cell in enumerate(cfg.cells):
        reporter.cell_start(cell.id, index, total)
        records.append(_run_cell(cfg, cell, shared_input, adapters, run_id, reference, reporter, ctx))
    reporter.run_done(records)
    return records


def _run_cell(
    cfg: BenchmarkConfig,
    cell: CellConfig,
    shared_input: SharedInput,
    adapters: dict[str, object],
    run_id: str,
    reference: np.ndarray,
    reporter: RunReporter,
    ctx: TelemetryContext,
) -> BenchmarkRecord:
    adapter = adapters[cell.backend]
    realized: RealizedConfig | None = None
    # Cell-scoped telemetry: thermal at entry, loadavg at entry, thermal at exit.
    # Probe failures degrade to None rather than aborting the cell.
    thermal_start = _safe_thermal_snapshot()
    loadavg_start = _safe_loadavg()
    try:
        realized = adapter.prepare(cell)
        reporter.cell_prepared(cell.id, realized.compute_unit)
        timing = run_timed_steps(
            adapter=adapter,
            latent=shared_input.latent,
            timestep=shared_input.timestep,
            text_embedding=shared_input.text_embedding,
            warmup=cfg.warmup,
            iterations=cfg.iterations,
            on_warmup=lambda i, total, ms: reporter.warmup_step(cell.id, i, total, ms),
            on_timed=lambda i, total, ms: reporter.timed_step(cell.id, i, total, ms),
        )
        reference_output = _reference_output(adapter, shared_input, reference)
        equivalence = compare_to_reference(
            timing.last_output,
            reference_output,
            mse_max=cfg.equivalence.mse_max,
            cosine_min=cfg.equivalence.cosine_min,
        )
        model_size = _model_size(adapter)
        thermal_end = _safe_thermal_snapshot()
        record = BenchmarkRecord(
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
            schema_version=ctx.schema_version,
            host_id_hash=ctx.host_id_hash,
            thermal_at_cell_start=_thermal_to_dict(thermal_start),
            thermal_at_cell_end=_thermal_to_dict(thermal_end),
            loadavg_at_cell_start=loadavg_start,
            env_vars_digest=ctx.env_vars_digest,
            power_sampler_interval_ms=ctx.power_sampler_interval_ms,
            latent_input_sha256=ctx.latent_input_sha256,
            text_embedding_input_sha256=ctx.text_embedding_input_sha256,
        )
        reporter.cell_done(record)
        return record
    except Exception as exc:
        reporter.cell_failed(cell.id, str(exc))
        return _failed_record(run_id, cell, realized, str(exc), ctx, thermal_start, loadavg_start)
    finally:
        adapter.teardown()


def _failed_record(
    run_id: str,
    cell: CellConfig,
    realized: RealizedConfig | None,
    reason: str,
    ctx: TelemetryContext,
    thermal_start: telemetry.ThermalSnapshot | None,
    loadavg_start: list[float] | None,
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
        schema_version=ctx.schema_version,
        host_id_hash=ctx.host_id_hash,
        thermal_at_cell_start=_thermal_to_dict(thermal_start),
        loadavg_at_cell_start=loadavg_start,
        env_vars_digest=ctx.env_vars_digest,
        power_sampler_interval_ms=ctx.power_sampler_interval_ms,
        latent_input_sha256=ctx.latent_input_sha256,
        text_embedding_input_sha256=ctx.text_embedding_input_sha256,
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


def _safe_thermal_snapshot() -> telemetry.ThermalSnapshot | None:
    try:
        return telemetry.collect_thermal_snapshot()
    except Exception:
        return None


def _safe_loadavg() -> list[float] | None:
    import os

    try:
        return list(os.getloadavg())
    except (OSError, AttributeError):
        return None


def _thermal_to_dict(snapshot: telemetry.ThermalSnapshot | None) -> dict | None:
    if snapshot is None:
        return None
    return {
        "cpu_speed_limit_pct": snapshot.cpu_speed_limit_pct,
        "throttled": snapshot.throttled,
        "source": snapshot.source,
        "detail": snapshot.detail,
    }
