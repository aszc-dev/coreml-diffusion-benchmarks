"""`sdbench power --session`: realign a whole multi-run session post-hoc.

The report validator reads sessions/<id>/aggregated.jsonl, which is computed
from the per-pass JSONLs — so the single-file `--input` mode does not fix a
multi-run bundle. This exercises the session mode end to end: per-pass plists in
a raw dir, each pass realigned and re-aggregated, tables regenerated."""

import plistlib
from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from sdbench.cli import app
from sdbench.results import BenchmarkRecord, load_jsonl, write_jsonl


def _record(run_id: str, cell_id: str, repeat_index: int, *, start: float, end: float) -> BenchmarkRecord:
    return BenchmarkRecord(
        run_id=run_id,
        cell_id=cell_id,
        backend="diffusers_mps",
        requested_compute_unit="MPS",
        realized_compute_unit="MPS",
        attention="NATIVE",
        precision="fp16",
        resolution=512,
        status="ok",
        latency_ms_median=100.0,
        latency_ms_iqr=2.0,
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
        active_window_start_s=start,
        active_window_end_s=end,
        repeat_index=repeat_index,
        repeat_count=3,
    )


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _write_plist(path, *, start_epoch: float, end_epoch: float, gpu_w: float) -> None:
    """A plist whose active window holds a steady gpu_w, idle 0 outside.

    Timestamps are absolute epoch seconds (the harness aligns plist <date> to the
    record's wall-clock window). One leaked full-power sample sits just inside the
    baseline window to prove the median (not the mean) is what gets reported."""
    docs = []
    # baseline before: idle, plus one leaked full-power sample at the edge
    for t in (start_epoch - 2.0, start_epoch - 1.5, start_epoch - 1.0, start_epoch - 0.5):
        docs.append((t, 0.0))
    docs.append((start_epoch - 0.2, gpu_w))  # leak — median must reject it
    # active: steady full power
    t = start_epoch
    while t < end_epoch:
        docs.append((t, gpu_w))
        t += 0.1
    # baseline after: idle
    for t in (end_epoch + 0.5, end_epoch + 1.0, end_epoch + 1.5):
        docs.append((t, 0.0))
    blob = b""
    for ts, w in docs:
        dt = _EPOCH + timedelta(seconds=ts)
        blob += plistlib.dumps({"timestamp": dt, "processor": {"gpu_power": w * 1000, "ane_power": 0}}) + b"\x00"
    path.write_bytes(blob)


def _matrix(tmp_path):
    cfg = tmp_path / "matrix.yaml"
    cfg.write_text(
        """
checkpoint: /tmp/sd15.safetensors
seed: 9
iterations: 10
warmup: 1
thermal: { throttle_policy: abort }
equivalence: { mse_max: 1.0e-3, cosine_min: 0.999 }
power: { interval_ms: 100, baseline_seconds: 2 }
cells:
  - { id: c, backend: diffusers_mps, compute_unit: MPS, attention: NATIVE, precision: fp16, resolution: 512 }
""",
        encoding="utf-8",
    )
    return cfg


def test_power_session_realigns_all_passes_and_reaggregates(tmp_path):
    session_dir = tmp_path / "sessions" / "sess1"
    session_dir.mkdir(parents=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Three passes, each its own plist with a steady 20 W active window. The
    # window timestamps are wall-clock; give each pass a distinct epoch offset.
    for i in range(3):
        rid = f"run{i}"
        # Distinct wall-clock epoch per pass; 1 s active window.
        start = 1_700_000_000.0 + 1000.0 * i
        end = start + 1.0
        rec = _record(rid, "c", i, start=start, end=end)
        write_jsonl([rec], session_dir / f"run-0{i}.jsonl")
        _write_plist(raw_dir / f"{rid}-powermetrics.plist", start_epoch=start, end_epoch=end, gpu_w=20.0)

    result = CliRunner().invoke(
        app,
        [
            "power",
            "--session", str(session_dir),
            "--raw-dir", str(raw_dir),
            "--config", str(_matrix(tmp_path)),
            "--output-dir", str(tmp_path / "tables"),
        ],
    )

    assert result.exit_code == 0, result.output

    # Each pass JSONL now carries the median power (20 W), not a leak-biased mean.
    for i in range(3):
        rec = load_jsonl(session_dir / f"run-0{i}.jsonl")[0]
        assert rec.gpu_power_w == 20.0
        # 20 W * 1 s window / 10 iterations = 2.0 J/step
        assert abs(rec.energy_per_unet_step_j - 2.0) < 1e-6

    # The aggregate the validator reads exists and has one median row per cell.
    agg = load_jsonl(session_dir / "aggregated.jsonl")
    assert len(agg) == 1
    assert abs(agg[0].energy_per_unet_step_j - 2.0) < 1e-6


def test_power_session_warns_on_missing_plist_but_keeps_pass(tmp_path):
    session_dir = tmp_path / "sessions" / "sess2"
    session_dir.mkdir(parents=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Pass exists, but its plist is absent.
    rec = _record("orphan", "c", 0, start=10.0, end=11.0)
    write_jsonl([rec], session_dir / "run-00.jsonl")

    result = CliRunner().invoke(
        app,
        [
            "power",
            "--session", str(session_dir),
            "--raw-dir", str(raw_dir),
            "--config", str(_matrix(tmp_path)),
            "--output-dir", str(tmp_path / "tables"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "missing" in result.output
    # Pass is left unchanged (power still None), not dropped.
    assert load_jsonl(session_dir / "run-00.jsonl")[0].gpu_power_w is None
