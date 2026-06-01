import plistlib
import time
from datetime import datetime, timedelta

import numpy as np

from sdbench.power import (
    PowerSample,
    apply_power_to_records,
    parse_powermetrics_plist,
    summarize_power,
)
from sdbench.results import BenchmarkRecord, load_jsonl, write_jsonl
from sdbench.timing import run_timed_steps


def _record(**overrides) -> BenchmarkRecord:
    base = dict(
        run_id="run",
        cell_id="cell",
        backend="coreml_diffusion",
        requested_compute_unit="CPU_AND_GPU",
        realized_compute_unit="CPU_AND_GPU",
        attention="ORIGINAL",
        precision="fp16",
        resolution=512,
        status="ok",
        latency_ms_median=100.0,
        latency_ms_iqr=5.0,
        gpu_power_w=None,
        ane_power_w=None,
        energy_per_unet_step_j=None,
        estimated_energy_per_50_step_image_j=None,
        mse=1e-7,
        cosine=0.9999,
        numerically_divergent=False,
        on_disk_size_bytes=100,
        weight_only_size_bytes=80,
        effective_bits_per_parameter=16.0,
        compute_precision="fp16",
        graph_capture_s=None,
        convert_s=None,
        first_load_compile_s=None,
        failure_reason=None,
    )
    base.update(overrides)
    return BenchmarkRecord(**base)


def _plist_doc(ts: datetime, *, gpu_mw, ane_mw, cpu_mw) -> bytes:
    # Power is reported under the `processor` dict in milliwatts on Apple Silicon.
    return plistlib.dumps(
        {"timestamp": ts, "processor": {"gpu_power": gpu_mw, "ane_power": ane_mw, "cpu_power": cpu_mw}}
    )


def test_parse_powermetrics_plist_reads_nul_separated_documents(tmp_path):
    t0 = datetime(2026, 5, 27, 12, 0, 0)
    stream = (
        _plist_doc(t0, gpu_mw=2000, ane_mw=500, cpu_mw=1000)
        + b"\x00"
        + _plist_doc(t0 + timedelta(seconds=1), gpu_mw=7000, ane_mw=3500, cpu_mw=1200)
        + b"\x00"
    )
    path = tmp_path / "power.plist"
    path.write_bytes(stream)

    samples = parse_powermetrics_plist(path)

    assert len(samples) == 2
    assert samples[0].gpu_w == 2.0 and samples[0].ane_w == 0.5 and samples[0].cpu_w == 1.0
    assert samples[1].gpu_w == 7.0 and samples[1].ane_w == 3.5
    assert samples[1].timestamp_s - samples[0].timestamp_s == 1.0


def test_parse_powermetrics_plist_is_defensive(tmp_path):
    t0 = datetime(2026, 5, 27, 12, 0, 0)
    stream = (
        _plist_doc(t0, gpu_mw=2000, ane_mw=500, cpu_mw=1000)
        + b"\x00"
        # missing ane channel -> 0.0
        + plistlib.dumps({"timestamp": t0 + timedelta(seconds=1), "processor": {"gpu_power": 3000}})
        + b"\x00"
        + b"<?xml truncated garbage"  # killed mid-write -> skipped
    )
    path = tmp_path / "power.plist"
    path.write_bytes(stream)

    samples = parse_powermetrics_plist(path)

    assert len(samples) == 2
    assert samples[1].gpu_w == 3.0
    assert samples[1].ane_w == 0.0


def test_apply_power_multi_cell_baseline_is_not_contaminated():
    # Two cells with disjoint windows. Each cell's active window carries high power;
    # everything else is idle. Without per-cell windowing, cell A's baseline would be
    # polluted by cell B's active window and its reported power would collapse.
    def block(times, gpu, ane):
        return [PowerSample(timestamp_s=t, cpu_w=1.0, gpu_w=gpu, ane_w=ane) for t in times]

    samples = (
        block([8.0, 8.5, 9.0, 9.5], 1.0, 0.5)       # A baseline before
        + block([10.0, 10.5, 11.0, 11.5], 10.0, 5.0)  # A active [10,12)
        + block([12.0, 12.5, 13.0, 13.5], 1.0, 0.5)   # A baseline after
        + block([18.0, 18.5, 19.0, 19.5], 1.0, 0.5)   # B baseline before
        + block([20.0, 20.5, 21.0, 21.5], 20.0, 8.0)  # B active [20,22)
        + block([22.0, 22.5, 23.0, 23.5], 1.0, 0.5)   # B baseline after
    )
    records = [
        _record(cell_id="A", active_window_start_s=10.0, active_window_end_s=12.0),
        _record(cell_id="B", active_window_start_s=20.0, active_window_end_s=22.0),
    ]

    updated = apply_power_to_records(records, samples, baseline_seconds=2.0, iterations=10)

    a, b = updated[0], updated[1]
    assert a.gpu_power_w == 9.0 and a.ane_power_w == 4.5
    assert b.gpu_power_w == 19.0 and b.ane_power_w == 7.5


def test_summarize_power_median_rejects_window_leak():
    # The window bounds are wall-clock-aligned across two processes, so a few
    # full-power active samples leak into the idle baseline and a few near-zero
    # ramp samples leak into the active window. A mean would read a contaminated
    # ~several-watt baseline and a depressed active average; the median rejects
    # both because each window is dominated by its steady state. Steady GPU is
    # 20 W, true idle is 0 W -> median net 20 W, mean net would be well under.
    def block(times, gpu, ane):
        return [PowerSample(timestamp_s=t, cpu_w=1.0, gpu_w=gpu, ane_w=ane) for t in times]

    # baseline window [8,10): mostly 0 W idle, but two leaked 20 W ramp samples
    baseline = block([8.0, 8.5, 9.0], 0.0, 0.0) + block([9.5, 9.9], 20.0, 0.0)
    # active window [10,12): mostly 20 W, but two leaked near-zero edge samples
    active = block([10.0, 10.2], 0.0, 0.0) + block([10.5, 11.0, 11.5], 20.0, 0.0)
    summary = summarize_power(baseline + active, 10.0, 12.0, timed_iterations=10)

    # median baseline = 0, median active = 20 -> net 20 W, unaffected by the leak
    assert summary.gpu_power_w == 20.0
    assert summary.energy_per_unet_step_j == 20.0 * (12.0 - 10.0) / 10


def test_summarize_power_even_count_median_averages_middle_two():
    samples = (
        [PowerSample(timestamp_s=t, cpu_w=1.0, gpu_w=0.0, ane_w=0.0) for t in (8.0, 9.0)]
        + [PowerSample(timestamp_s=10.0, cpu_w=1.0, gpu_w=10.0, ane_w=0.0),
           PowerSample(timestamp_s=11.0, cpu_w=1.0, gpu_w=20.0, ane_w=0.0)]
    )
    summary = summarize_power(samples, 10.0, 12.0, timed_iterations=10)

    assert summary.gpu_power_w == 15.0  # (10 + 20) / 2, idle baseline 0


def test_apply_power_skips_failed_and_missing_window_records():
    samples = [PowerSample(timestamp_s=10.5, cpu_w=1.0, gpu_w=10.0, ane_w=5.0)]
    failed = _record(cell_id="f", status="failed", active_window_start_s=None, active_window_end_s=None)
    no_window = _record(cell_id="nw", active_window_start_s=None, active_window_end_s=None)

    updated = apply_power_to_records([failed, no_window], samples, baseline_seconds=2.0, iterations=10)

    assert all(r.gpu_power_w is None and r.ane_power_w is None for r in updated)


def test_apply_power_round_trips_through_jsonl(tmp_path):
    samples = (
        [PowerSample(timestamp_s=t, cpu_w=1.0, gpu_w=1.0, ane_w=0.5) for t in (8.0, 9.0)]
        + [PowerSample(timestamp_s=t, cpu_w=1.0, gpu_w=10.0, ane_w=5.0) for t in (10.0, 11.0)]
        + [PowerSample(timestamp_s=t, cpu_w=1.0, gpu_w=1.0, ane_w=0.5) for t in (12.0, 13.0)]
    )
    record = _record(active_window_start_s=10.0, active_window_end_s=12.0)
    updated = apply_power_to_records([record], samples, baseline_seconds=2.0, iterations=10)

    path = tmp_path / "results.jsonl"
    write_jsonl(updated, path)
    loaded = load_jsonl(path)

    assert loaded[0].gpu_power_w == 9.0
    assert loaded[0].ane_power_w == 4.5
    assert loaded[0].energy_per_unet_step_j is not None


def test_run_timed_steps_records_wall_window():
    class StepAdapter:
        def step(self, latent, timestep, text_embedding):
            return latent

    latent = np.zeros((1, 4, 8, 8), dtype=np.float32)
    before = time.time()
    result = run_timed_steps(
        adapter=StepAdapter(),
        latent=latent,
        timestep=1,
        text_embedding=np.zeros((1, 77, 768), dtype=np.float32),
        warmup=1,
        iterations=10,
    )
    after = time.time()

    assert result.active_wall_start_s is not None and result.active_wall_end_s is not None
    assert result.active_wall_start_s <= result.active_wall_end_s
    assert before <= result.active_wall_start_s
    assert result.active_wall_end_s <= after
