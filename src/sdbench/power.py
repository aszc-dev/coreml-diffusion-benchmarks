import plistlib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from sdbench.results import BenchmarkRecord


@dataclass(frozen=True)
class PowerSample:
    timestamp_s: float
    cpu_w: float
    gpu_w: float
    ane_w: float


@dataclass(frozen=True)
class PowerSummary:
    gpu_power_w: float
    ane_power_w: float
    energy_per_unet_step_j: float
    estimated_energy_per_50_step_image_j: float


def summarize_power(
    samples: list[PowerSample],
    active_start_s: float,
    active_end_s: float,
    timed_iterations: int,
) -> PowerSummary:
    # Per-engine power is the *median* sample over each window, not the mean
    # (R6.4). The harness and the sampler are separate processes aligned only by
    # wall clock, so the recorded ``[start, end)`` bounds drift by up to a sample
    # interval relative to the actual GPU ramp: a handful of full-power (~20 W)
    # samples leak into the idle baseline window, and a handful of near-zero
    # samples leak into the active window. The mean inherits that contamination —
    # an idle baseline whose true level is 0 W reads 2–8 W once a few leaked
    # active samples are averaged in, which is then subtracted from active and
    # collapses energy/step by up to 50% pass-to-pass (observed spread 0.32–0.77
    # on an otherwise-idle host). The median rejects those edge samples because
    # the window is dominated by the steady state on both sides, and it is the
    # right statistic for a *per-step steady-state* cost anyway: we want the
    # settled cost of one UNet step, not one amortised over the engine's
    # spin-up/down ramp. This mirrors R5's "report median latency, never mean".
    active = [sample for sample in samples if active_start_s <= sample.timestamp_s < active_end_s]
    baseline = [sample for sample in samples if sample.timestamp_s < active_start_s or sample.timestamp_s >= active_end_s]
    if not active:
        return PowerSummary(0.0, 0.0, 0.0, 0.0)

    baseline_gpu = _median([sample.gpu_w for sample in baseline])
    baseline_ane = _median([sample.ane_w for sample in baseline])
    gpu_power = max(0.0, _median([sample.gpu_w for sample in active]) - baseline_gpu)
    ane_power = max(0.0, _median([sample.ane_w for sample in active]) - baseline_ane)
    duration = active_end_s - active_start_s
    total_energy = (gpu_power + ane_power) * duration
    per_step = total_energy / timed_iterations if timed_iterations else 0.0
    return PowerSummary(
        gpu_power_w=gpu_power,
        ane_power_w=ane_power,
        energy_per_unet_step_j=per_step,
        estimated_energy_per_50_step_image_j=per_step * 50.0,
    )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def parse_powermetrics_plist(path: str | Path) -> list[PowerSample]:
    """Parse a `powermetrics -f plist` capture into PowerSamples.

    powermetrics streams one plist document per sampling interval, the documents
    concatenated and separated by a NUL byte (R6.1). Per-engine power is reported
    in milliwatts, either at the top level or under a `processor` dict depending on
    the macOS version. Timestamps are wall-clock, used for post-hoc alignment (R6.3)."""
    raw = Path(path).read_bytes()
    samples: list[PowerSample] = []
    for fragment in raw.split(b"\x00"):
        fragment = fragment.strip()
        if not fragment:
            continue
        try:
            doc = plistlib.loads(fragment)
        except Exception:
            # A truncated final fragment is expected when powermetrics is killed mid-write.
            continue
        timestamp_s = _sample_timestamp(doc)
        if timestamp_s is None:
            continue
        processor = doc.get("processor", {}) if isinstance(doc, dict) else {}
        samples.append(
            PowerSample(
                timestamp_s=timestamp_s,
                cpu_w=_milliwatts_to_watts(doc, processor, "cpu_power"),
                gpu_w=_milliwatts_to_watts(doc, processor, "gpu_power"),
                ane_w=_milliwatts_to_watts(doc, processor, "ane_power"),
            )
        )
    samples.sort(key=lambda sample: sample.timestamp_s)
    return samples


def apply_power_to_records(
    records: list[BenchmarkRecord],
    samples: list[PowerSample],
    baseline_seconds: float,
    iterations: int,
) -> list[BenchmarkRecord]:
    """Fill per-cell power fields from a full-run sample stream, post-hoc.

    For each successful cell, only samples within `baseline_seconds` of that cell's
    own timed window are considered, so other cells' active windows are NOT mistaken
    for this cell's idle baseline (R6.2). Cells without a recorded window (failed, or
    pre-power records) pass through unchanged."""
    updated: list[BenchmarkRecord] = []
    for record in records:
        start = record.active_window_start_s
        end = record.active_window_end_s
        if record.status != "ok" or start is None or end is None:
            updated.append(record)
            continue
        windowed = [s for s in samples if (start - baseline_seconds) <= s.timestamp_s < (end + baseline_seconds)]
        summary = summarize_power(windowed, start, end, iterations)
        updated.append(
            replace(
                record,
                gpu_power_w=summary.gpu_power_w,
                ane_power_w=summary.ane_power_w,
                energy_per_unet_step_j=summary.energy_per_unet_step_j,
                estimated_energy_per_50_step_image_j=summary.estimated_energy_per_50_step_image_j,
            )
        )
    return updated


def _milliwatts_to_watts(doc, processor, key: str) -> float:
    value = doc.get(key) if isinstance(doc, dict) else None
    if value is None and isinstance(processor, dict):
        value = processor.get(key)
    try:
        return float(value) / 1000.0 if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _sample_timestamp(doc) -> float | None:
    if not isinstance(doc, dict):
        return None
    value = doc.get("timestamp")
    if isinstance(value, datetime):
        # plist <date> is UTC ("...Z") but plistlib returns a naive datetime;
        # without pinning UTC, .timestamp() would shift by the local offset and
        # the samples would no longer align to the harness's epoch window.
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    return None
