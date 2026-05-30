"""Per-cell aggregation across the N passes of a ``--repeats N`` session.

A *session* is N independent matrix passes that share a ``session_id`` and were
deliberately run with a thermal cooldown between them so each pass measures a
comparable cold start. This module reduces those N passes to one row per cell:
the median of latency / GPU power / ANE power / energy/step, plus the p10–p90
between-run spread — the statistic that tells the reader whether the cell's
energy figure is reproducible or chasing system noise (R5.4, R6.5).

Outputs land at ``results/data/sessions/<session_id>/aggregated.jsonl`` (one row
per cell) and a sibling ``session.json`` that lists every pass's ``run_id``,
``repeat_index`` and its environment-manifest path so the aggregate can be
traced back to its raw passes.
"""

import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median

from sdbench.results import BenchmarkRecord


def aggregate_session(records: list[BenchmarkRecord]) -> list[BenchmarkRecord]:
    """Reduce N passes of the same matrix to one ``BenchmarkRecord`` per cell.

    Aggregation rules per field:

    * **Median + p10/p90** for the metrics whose between-run variance is the
      whole point of running multi-run: ``latency_ms_median``, ``gpu_power_w``,
      ``ane_power_w``, ``energy_per_unet_step_j``.
    * **Median only** for ``mse`` / ``cosine`` (expected stable across passes;
      a real shift here would be a determinism bug, not a noise floor).
    * **Identical-across-passes** invariants (sizing, conversion timings,
      attention/precision/resolution, ``compute_precision``, ``schema_version``,
      ``provenance_digest``, ``latent_input_sha256``,
      ``text_embedding_input_sha256``): asserted to match across all ``ok``
      passes; first-pass value is carried forward. A mismatch surfaces as a
      ``failure_reason`` on the aggregated row instead of silently picking one.
    * **Status**: ``ok`` if every pass succeeded, ``degraded`` if some passes
      failed but at least one succeeded, ``failed`` if every pass failed. The
      original ``numerically_divergent`` flag is propagated as ``True`` if *any*
      pass tripped it (a flaky equivalence is still flagged).

    The returned rows carry ``repeat_index=None`` and ``repeat_count=N`` — they
    are the *session* row, not a member of it; pre-aggregate per-pass rows stay
    under ``sessions/<id>/run-NN.jsonl`` for audit. The keying contract of
    :func:`sdbench.results.upsert_jsonl` (``(cell_id, repeat_index or 0)``)
    keeps these from colliding with per-pass rows in any shared file.
    """
    if not records:
        return []
    by_cell: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        by_cell[record.cell_id].append(record)
    out: list[BenchmarkRecord] = []
    for cell_id, group in by_cell.items():
        out.append(_aggregate_one_cell(cell_id, group))
    return out


def _aggregate_one_cell(cell_id: str, group: list[BenchmarkRecord]) -> BenchmarkRecord:
    ok = [r for r in group if r.status == "ok"]
    failed = [r for r in group if r.status != "ok"]
    n_runs_ok = len(ok)
    n_runs_failed = len(failed)
    # If no pass succeeded we still emit a row so the reader sees the cell
    # didn't drop out silently; latency/power stay None and the failure reason
    # is the most recent one.
    if not ok:
        anchor = group[0]
        return replace(
            anchor,
            run_id=anchor.run_id,
            repeat_index=None,
            repeat_count=len(group),
            status="failed",
            failure_reason=group[-1].failure_reason or anchor.failure_reason,
            n_runs_ok=0,
            n_runs_failed=n_runs_failed,
        )

    anchor = ok[0]
    invariant_mismatches = _check_invariants(ok)
    median_lat = _median_or_none([r.latency_ms_median for r in ok])
    median_gpu = _median_or_none([r.gpu_power_w for r in ok])
    median_ane = _median_or_none([r.ane_power_w for r in ok])
    median_energy = _median_or_none([r.energy_per_unet_step_j for r in ok])
    median_50step = _median_or_none([r.estimated_energy_per_50_step_image_j for r in ok])
    median_mse = _median_or_none([r.mse for r in ok])
    median_cos = _median_or_none([r.cosine for r in ok])
    lat_p10, lat_p90 = _p10_p90([r.latency_ms_median for r in ok])
    gpu_p10, gpu_p90 = _p10_p90([r.gpu_power_w for r in ok])
    ane_p10, ane_p90 = _p10_p90([r.ane_power_w for r in ok])
    energy_p10, energy_p90 = _p10_p90([r.energy_per_unet_step_j for r in ok])
    # Between-run IQR on latency: keeps the single-cell ``latency_ms_iqr``
    # (within-pass dispersion) faithfully, but exposes the across-pass spread
    # too so a stable-within / wandering-across cell can be spotted in one
    # column.
    median_within_iqr = _median_or_none([r.latency_ms_iqr for r in ok])

    status = "ok" if not failed else "degraded"
    failure_reason = "; ".join(invariant_mismatches) if invariant_mismatches else None
    if invariant_mismatches:
        # An invariant mismatch is a harness-side correctness issue, not a
        # metrology one — surface it as the row's failure_reason even if the
        # passes themselves succeeded.
        status = "degraded"

    return BenchmarkRecord(
        run_id=anchor.run_id,
        cell_id=cell_id,
        backend=anchor.backend,
        requested_compute_unit=anchor.requested_compute_unit,
        realized_compute_unit=anchor.realized_compute_unit,
        attention=anchor.attention,
        precision=anchor.precision,
        resolution=anchor.resolution,
        status=status,
        latency_ms_median=median_lat,
        latency_ms_iqr=median_within_iqr,
        gpu_power_w=median_gpu,
        ane_power_w=median_ane,
        energy_per_unet_step_j=median_energy,
        estimated_energy_per_50_step_image_j=median_50step,
        mse=median_mse,
        cosine=median_cos,
        numerically_divergent=any(bool(r.numerically_divergent) for r in ok),
        on_disk_size_bytes=anchor.on_disk_size_bytes,
        weight_only_size_bytes=anchor.weight_only_size_bytes,
        effective_bits_per_parameter=anchor.effective_bits_per_parameter,
        compute_precision=anchor.compute_precision,
        graph_capture_s=anchor.graph_capture_s,
        convert_s=anchor.convert_s,
        first_load_compile_s=anchor.first_load_compile_s,
        failure_reason=failure_reason,
        active_window_start_s=None,
        active_window_end_s=None,
        provenance_digest=anchor.provenance_digest,
        schema_version=anchor.schema_version,
        host_id_hash=anchor.host_id_hash,
        thermal_at_cell_start=anchor.thermal_at_cell_start,
        thermal_at_cell_end=anchor.thermal_at_cell_end,
        loadavg_at_cell_start=anchor.loadavg_at_cell_start,
        env_vars_digest=anchor.env_vars_digest,
        power_sampler_interval_ms=anchor.power_sampler_interval_ms,
        latent_input_sha256=anchor.latent_input_sha256,
        text_embedding_input_sha256=anchor.text_embedding_input_sha256,
        session_id=anchor.session_id,
        repeat_index=None,
        repeat_count=len(group),
        latency_ms_p10=lat_p10,
        latency_ms_p90=lat_p90,
        gpu_power_w_p10=gpu_p10,
        gpu_power_w_p90=gpu_p90,
        ane_power_w_p10=ane_p10,
        ane_power_w_p90=ane_p90,
        energy_per_unet_step_j_p10=energy_p10,
        energy_per_unet_step_j_p90=energy_p90,
        n_runs_ok=n_runs_ok,
        n_runs_failed=n_runs_failed,
    )


# Fields that must be identical across all ``ok`` passes of a cell; otherwise
# the harness produced incoherent inputs and the aggregate would lie about
# what was measured. Reported as the aggregated row's ``failure_reason``.
_INVARIANT_FIELDS: tuple[str, ...] = (
    "backend",
    "requested_compute_unit",
    "realized_compute_unit",
    "attention",
    "precision",
    "resolution",
    "compute_precision",
    "on_disk_size_bytes",
    "weight_only_size_bytes",
    "effective_bits_per_parameter",
    "schema_version",
    "provenance_digest",
    "latent_input_sha256",
    "text_embedding_input_sha256",
    "graph_capture_s",
    "convert_s",
    "first_load_compile_s",
)


def _check_invariants(ok: list[BenchmarkRecord]) -> list[str]:
    if len(ok) < 2:
        return []
    issues: list[str] = []
    anchor = ok[0]
    for field in _INVARIANT_FIELDS:
        anchor_val = getattr(anchor, field)
        for other in ok[1:]:
            other_val = getattr(other, field)
            if anchor_val != other_val:
                issues.append(f"invariant '{field}' mismatch across passes: {anchor_val!r} vs {other_val!r}")
                break
    return issues


def _median_or_none(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(median(clean))


def _p10_p90(values: list[float | None]) -> tuple[float | None, float | None]:
    """Linearly-interpolated 10th and 90th percentiles, or ``(None, None)``.

    Uses the standard "exclusive" definition consistent with NumPy's default so
    a session of 5 passes lands its p10/p90 at fractional indices rather than
    collapsing to min/max. With fewer than 3 valid samples we have nothing to
    say about the spread and return ``(None, None)`` — the report layer reads
    that as "show median only".
    """
    clean = sorted(v for v in values if v is not None)
    if len(clean) < 3:
        return (None, None)
    return (_percentile(clean, 0.10), _percentile(clean, 0.90))


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile on empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def write_session_manifest(
    path: Path,
    *,
    session_id: str,
    repeats: int,
    cooldown_s: float,
    passes: list[list[BenchmarkRecord]],
) -> None:
    """Write a compact session manifest listing every pass's run_id and counts.

    The bundle's contributor README points readers at ``aggregated.jsonl``, but
    a maintainer auditing reproducibility needs the per-pass linkage:
    ``run_id``, how many cells were ``ok`` / ``failed``, and which session this
    aggregation belongs to. Keep it small (no environment duplication — those
    live in ``results/data/environments/<run_id>.json``)."""
    payload = {
        "session_id": session_id,
        "repeats": repeats,
        "cooldown_s": cooldown_s,
        "passes": [
            {
                "repeat_index": (records[0].repeat_index if records else idx),
                "run_id": (records[0].run_id if records else None),
                "n_records": len(records),
                "n_ok": sum(1 for r in records if r.status == "ok"),
                "n_failed": sum(1 for r in records if r.status != "ok"),
            }
            for idx, records in enumerate(passes)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_aggregated(path: Path) -> list[BenchmarkRecord]:
    """Convenience reader for ``aggregated.jsonl`` — mirrors ``load_jsonl`` but
    is named so report.py can switch on intent rather than path."""
    from sdbench.results import load_jsonl

    return load_jsonl(path)


def is_aggregated_row(record: BenchmarkRecord) -> bool:
    """A row is the session-level aggregate when its ``repeat_count`` is set
    but its ``repeat_index`` is None — i.e. it summarises ``repeat_count``
    passes rather than being one of them."""
    return record.repeat_count is not None and record.repeat_index is None


# asdict re-export so callers that already imported aggregate-side helpers
# don't have to import dataclasses just to serialise a row.
def to_dict(record: BenchmarkRecord) -> dict:
    return asdict(record)
