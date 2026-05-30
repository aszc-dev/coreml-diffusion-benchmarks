"""Aggregator + multi-run upsert tests.

These exercise three contracts: per-cell reduction (median + p10/p90 over the
passes), invariant guarding (mismatched sizing/precision across passes must
surface as a failure_reason rather than be silently averaged away), and the
upsert key extension that lets per-pass rows coexist in one JSONL without
clobbering each other.
"""

import json

from sdbench.aggregate import (
    aggregate_session,
    is_aggregated_row,
    write_session_manifest,
)
from sdbench.results import BenchmarkRecord, load_jsonl, upsert_jsonl


# The floor :mod:`sdbench.report` enforces on multi-run bundles. Kept here so
# the test fails loudly if either side drifts away from the other — three is
# the smallest sample at which the p10/p90 interpolation is well-defined.
SESSION_MIN_RUNS_OK = 3


def _make_record(
    *,
    cell_id: str,
    repeat_index: int,
    latency_ms_median: float,
    energy: float | None,
    gpu_w: float | None = None,
    ane_w: float | None = None,
    status: str = "ok",
    repeat_count: int = 5,
    session_id: str = "session-1",
    run_id: str | None = None,
    backend: str = "diffusers_mps",
    compute_unit: str = "MPS",
    attention: str = "NATIVE",
    precision: str = "fp16",
    resolution: int = 512,
    on_disk: int | None = 1024,
    failure_reason: str | None = None,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        run_id=run_id or f"run-{repeat_index}",
        cell_id=cell_id,
        backend=backend,
        requested_compute_unit=compute_unit,
        realized_compute_unit=compute_unit,
        attention=attention,
        precision=precision,
        resolution=resolution,
        status=status,
        latency_ms_median=latency_ms_median if status == "ok" else None,
        latency_ms_iqr=0.5 if status == "ok" else None,
        gpu_power_w=gpu_w,
        ane_power_w=ane_w,
        energy_per_unet_step_j=energy,
        estimated_energy_per_50_step_image_j=(energy * 50) if energy is not None else None,
        mse=1.0e-5 if status == "ok" else None,
        cosine=0.9999 if status == "ok" else None,
        numerically_divergent=False if status == "ok" else None,
        on_disk_size_bytes=on_disk,
        weight_only_size_bytes=on_disk,
        effective_bits_per_parameter=16.0,
        compute_precision=precision,
        graph_capture_s=None,
        convert_s=None,
        first_load_compile_s=None,
        failure_reason=failure_reason,
        session_id=session_id,
        repeat_index=repeat_index,
        repeat_count=repeat_count,
    )


def test_aggregator_computes_median_and_spread_per_cell():
    # Five passes with energy drifting between 1.0 and 1.9 J — exactly the
    # kind of between-run noise the multi-run mode is designed to expose.
    energies = [1.0, 1.2, 1.4, 1.7, 1.9]
    latencies = [400.0, 402.0, 401.0, 403.0, 399.0]
    records = [
        _make_record(
            cell_id="ours-ane-fp16",
            repeat_index=i,
            latency_ms_median=latencies[i],
            energy=energies[i],
            gpu_w=0.1,
            ane_w=2.0 + i * 0.1,
        )
        for i in range(5)
    ]

    [agg] = aggregate_session(records)

    assert agg.cell_id == "ours-ane-fp16"
    assert agg.status == "ok"
    assert agg.n_runs_ok == 5
    assert agg.n_runs_failed == 0
    assert agg.repeat_index is None
    assert agg.repeat_count == 5
    # Median of 5 sorted values is the middle one.
    assert agg.energy_per_unet_step_j == 1.4
    assert agg.latency_ms_median == 401.0
    # p10/p90 must bracket the median and stay inside the observed range.
    assert agg.energy_per_unet_step_j_p10 is not None
    assert agg.energy_per_unet_step_j_p90 is not None
    assert 1.0 <= agg.energy_per_unet_step_j_p10 < 1.4 < agg.energy_per_unet_step_j_p90 <= 1.9
    assert agg.gpu_power_w == 0.1  # constant across passes
    assert is_aggregated_row(agg)


def test_aggregator_below_three_passes_skips_spread_but_keeps_median():
    # With only two ok passes the p10/p90 interpolation is undefined; the
    # aggregator should return medians but None for the spread columns so the
    # report layer can surface "spread unavailable" rather than fake bounds.
    records = [
        _make_record(cell_id="cell", repeat_index=0, latency_ms_median=400.0, energy=1.0),
        _make_record(cell_id="cell", repeat_index=1, latency_ms_median=410.0, energy=1.2),
    ]
    [agg] = aggregate_session(records)
    assert agg.n_runs_ok == 2
    assert agg.energy_per_unet_step_j == 1.1  # median of two values
    assert agg.energy_per_unet_step_j_p10 is None
    assert agg.energy_per_unet_step_j_p90 is None


def test_aggregator_marks_partial_session_as_degraded():
    # One pass failed (e.g. transient OOM). The aggregate should still publish
    # the medians from the surviving passes but mark the row as degraded so
    # validate-report can decide what to do with it.
    records = [
        _make_record(cell_id="cell", repeat_index=0, latency_ms_median=400.0, energy=1.0),
        _make_record(cell_id="cell", repeat_index=1, latency_ms_median=410.0, energy=1.2),
        _make_record(
            cell_id="cell",
            repeat_index=2,
            latency_ms_median=0,
            energy=None,
            status="failed",
            failure_reason="oom",
        ),
    ]
    [agg] = aggregate_session(records)
    assert agg.n_runs_ok == 2
    assert agg.n_runs_failed == 1
    assert agg.status == "degraded"


def test_aggregator_all_passes_failed_keeps_row_as_failed():
    records = [
        _make_record(
            cell_id="mlx-gpu-q8",
            repeat_index=i,
            latency_ms_median=0,
            energy=None,
            status="failed",
            failure_reason="unsupported quant path",
        )
        for i in range(3)
    ]
    [agg] = aggregate_session(records)
    assert agg.status == "failed"
    assert agg.n_runs_ok == 0
    assert agg.n_runs_failed == 3
    assert "unsupported quant path" in (agg.failure_reason or "")


def test_aggregator_flags_invariant_mismatch_across_passes():
    # Sizing changed mid-session — that means the harness somehow loaded a
    # different artifact between passes; averaging would lie about what was
    # measured. Surface it as a failure_reason instead.
    records = [
        _make_record(cell_id="cell", repeat_index=0, latency_ms_median=400.0, energy=1.0, on_disk=1024),
        _make_record(cell_id="cell", repeat_index=1, latency_ms_median=410.0, energy=1.2, on_disk=2048),
        _make_record(cell_id="cell", repeat_index=2, latency_ms_median=405.0, energy=1.1, on_disk=1024),
    ]
    [agg] = aggregate_session(records)
    assert agg.status == "degraded"
    assert agg.failure_reason is not None
    assert "on_disk_size_bytes" in agg.failure_reason


def test_aggregator_groups_independent_cells_independently():
    a = [
        _make_record(cell_id="A", repeat_index=i, latency_ms_median=100.0 + i, energy=0.5)
        for i in range(3)
    ]
    b = [
        _make_record(cell_id="B", repeat_index=i, latency_ms_median=200.0 + i, energy=1.5)
        for i in range(3)
    ]
    agg = {row.cell_id: row for row in aggregate_session(a + b)}
    assert set(agg) == {"A", "B"}
    assert agg["A"].latency_ms_median == 101.0
    assert agg["B"].latency_ms_median == 201.0


def test_upsert_jsonl_keys_by_cell_id_and_repeat_index(tmp_path):
    path = tmp_path / "results.jsonl"
    pass0 = [
        _make_record(cell_id="cell", repeat_index=0, latency_ms_median=100.0, energy=1.0),
    ]
    pass1 = [
        _make_record(cell_id="cell", repeat_index=1, latency_ms_median=110.0, energy=1.1),
    ]
    upsert_jsonl(pass0, path)
    upsert_jsonl(pass1, path)
    rows = load_jsonl(path)
    # Both passes must coexist — pre-v3 upsert keyed by cell_id alone, which
    # silently overwrote the first pass. Regression guard.
    assert sorted(r.repeat_index for r in rows if r.repeat_index is not None) == [0, 1]
    # Re-running pass 0 must overwrite *only* its slot.
    updated = [_make_record(cell_id="cell", repeat_index=0, latency_ms_median=999.0, energy=9.9)]
    upsert_jsonl(updated, path)
    rows = load_jsonl(path)
    by_idx = {r.repeat_index: r for r in rows}
    assert by_idx[0].latency_ms_median == 999.0
    assert by_idx[1].latency_ms_median == 110.0


def test_write_session_manifest_lists_every_pass(tmp_path):
    passes = [
        [_make_record(cell_id="cell", repeat_index=i, latency_ms_median=100.0, energy=1.0, run_id=f"run-{i}")]
        for i in range(3)
    ]
    path = tmp_path / "session.json"
    write_session_manifest(path, session_id="sess", repeats=3, cooldown_s=30.0, passes=passes)
    payload = json.loads(path.read_text())
    assert payload["session_id"] == "sess"
    assert payload["repeats"] == 3
    assert [p["run_id"] for p in payload["passes"]] == ["run-0", "run-1", "run-2"]
    assert all(p["n_ok"] == 1 and p["n_failed"] == 0 for p in payload["passes"])


def test_session_min_runs_ok_floor_is_three():
    """Guard that ``validate-report``'s floor stays at three.

    The aggregator only emits p10/p90 when it has ≥3 ok passes; lowering this
    floor in :mod:`sdbench.report` without also lowering the aggregator's
    interpolation threshold would let bundles pass validation with empty
    spread columns. Keep the two in lockstep here."""
    from sdbench.report import SESSION_MIN_RUNS_OK as floor

    assert floor == SESSION_MIN_RUNS_OK == 3
