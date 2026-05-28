from pathlib import Path

import numpy as np

from sdbench.adapter import RealizedConfig
from sdbench.config import (
    BenchmarkConfig,
    CellConfig,
    EquivalenceConfig,
    PowerConfig,
    ThermalConfig,
)
from sdbench.inputs import SharedInput
from sdbench.orchestrator import run_matrix
from sdbench.progress import RunReporter
from sdbench.results import load_jsonl
from sdbench.tui.power_session import PowerSession, authorize_sudo, caffeinate_command, powermetrics_command
from sdbench.tui.run_view import _equivalence
from sdbench.tui.run_cmd import run_benchmark
from sdbench.tui.workspace import Workspace


class PassingAdapter:
    name = "passing"

    def prepare(self, cfg):
        return RealizedConfig(compute_unit=cfg.compute_unit, attention=cfg.attention, precision=cfg.precision, artifact_paths=[])

    def step(self, latent, timestep, text_embedding):
        return latent

    def teardown(self):
        pass


class FailingAdapter(PassingAdapter):
    def prepare(self, cfg):
        raise RuntimeError("boom")


# ----- reporter hooks -----

class RecordingReporter(RunReporter):
    def __init__(self):
        self.events = []

    def run_start(self, total_cells):
        self.events.append(("run_start", total_cells))

    def cell_start(self, cell_id, index, total):
        self.events.append(("cell_start", cell_id))

    def cell_prepared(self, cell_id, compute_unit):
        self.events.append(("prepared", cell_id))

    def timed_step(self, cell_id, index, total, latency_ms):
        self.events.append(("timed", cell_id, index))

    def cell_done(self, record):
        self.events.append(("done", record.cell_id))

    def cell_failed(self, cell_id, reason):
        self.events.append(("failed", cell_id))

    def run_done(self, records):
        self.events.append(("run_done", len(records)))


def _cfg(tmp_path):
    return BenchmarkConfig(
        checkpoint=tmp_path / "sd15.safetensors",
        seed=0, iterations=10, warmup=1,
        thermal=ThermalConfig(throttle_policy="flag", abort_on_throttle=False),
        equivalence=EquivalenceConfig(mse_max=1e-3, cosine_min=0.999),
        power=PowerConfig(interval_ms=100, baseline_seconds=2),
        cells=[
            CellConfig(id="fail", backend="failing", compute_unit="GPU", attention="NATIVE", precision="fp16", resolution=512),
            CellConfig(id="pass", backend="passing", compute_unit="GPU", attention="NATIVE", precision="fp16", resolution=512),
        ],
    )


def test_reporter_receives_lifecycle_events(tmp_path):
    shared = SharedInput(latent=np.zeros((1, 1), np.float32), timestep=1, text_embedding=np.zeros((1, 1), np.float32))
    reporter = RecordingReporter()

    run_matrix(_cfg(tmp_path), shared, {"failing": FailingAdapter(), "passing": PassingAdapter()}, "r", tmp_path, reporter=reporter)

    kinds = [e[0] for e in reporter.events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_done"
    assert ("failed", "fail") in reporter.events
    assert ("prepared", "pass") in reporter.events
    assert ("done", "pass") in reporter.events
    assert reporter.events.count(("timed", "pass", 10)) == 1  # 10 timed iterations reported


# ----- power session command construction -----

def test_caffeinate_binds_pid():
    assert caffeinate_command(123) == ["caffeinate", "-dimsu", "-w", "123"]


def test_powermetrics_command_runs_under_sudo_non_interactive():
    # `-n` keeps sudo from ever touching /dev/tty (writes from sudo / powermetrics
    # to the controlling terminal corrupt the fullscreen Live — regression guard).
    cmd = powermetrics_command("/tmp/x.plist", 100, ["cpu_power", "gpu_power"])
    assert cmd[:3] == ["sudo", "-n", "powermetrics"]
    assert "cpu_power,gpu_power" in cmd


def test_default_spawn_isolates_subprocess_from_controlling_tty(monkeypatch):
    # The sampler / caffeinate MUST be detached from our session and have stdin
    # closed — otherwise the child can write to /dev/tty and bypass every
    # stdout/stderr redirect we install. This regression-guards the actual fix
    # for the "C-c on the run fixes the screen but the run keeps going" bug.
    import subprocess

    from sdbench.tui import power_session

    captured = {}

    class _FakePopen:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(power_session.subprocess, "Popen", _FakePopen)
    power_session._default_spawn(["caffeinate"])

    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["stdout"] is subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is subprocess.DEVNULL


class _FakeProc:
    def terminate(self):
        pass

    def wait(self, timeout=None):
        pass

    def kill(self):
        pass


def test_power_session_only_caffeinates_when_disabled(tmp_path):
    spawned = []
    with PowerSession(tmp_path / "x.plist", 100, ["cpu_power"], enabled=False, spawn=lambda a: spawned.append(a) or _FakeProc()):
        pass
    assert len(spawned) == 1 and spawned[0][0] == "caffeinate"


def test_power_session_spawns_sampler_when_enabled(tmp_path):
    spawned = []
    with PowerSession(tmp_path / "x.plist", 50, ["gpu_power"], enabled=True, spawn=lambda a: spawned.append(a) or _FakeProc()):
        pass
    assert any(a[0] == "caffeinate" for a in spawned)
    assert any("powermetrics" in a for a in spawned)


def test_authorize_sudo_reports_outcome():
    from types import SimpleNamespace

    assert authorize_sudo(runner=lambda cmd: SimpleNamespace(returncode=0)) is True
    assert authorize_sudo(runner=lambda cmd: SimpleNamespace(returncode=1)) is False

    def boom(cmd):
        raise FileNotFoundError

    assert authorize_sudo(runner=boom) is False


# ----- equivalence is shown as a value, not just a label -----

def test_equivalence_flagged_shows_cosine_and_mse():
    from types import SimpleNamespace

    out = _equivalence(SimpleNamespace(cosine=0.9969, mse=5.6e-3, numerically_divergent=True))
    assert "cos 0.9969" in out and "flagged" in out and "mse" in out


def test_equivalence_match_shows_cosine_without_flag():
    from types import SimpleNamespace

    out = _equivalence(SimpleNamespace(cosine=1.0, mse=0.0, numerically_divergent=False))
    assert "cos 1.0000" in out and "flagged" not in out


def test_equivalence_absent_is_empty():
    from types import SimpleNamespace

    assert _equivalence(SimpleNamespace(cosine=None, mse=None, numerically_divergent=None)) == ""


# ----- run_benchmark end-to-end (fakes, no power) -----

def _write_matrix(tmp_path) -> Path:
    path = tmp_path / "matrix.yaml"
    path.write_text(
        f"""
checkpoint: {tmp_path}/missing.safetensors
seed: 0
warmup: 1
iterations: 10
resolution_default: 512
equivalence:
  mse_max: 1.0e-3
  cosine_min: 0.999
power:
  interval_ms: 100
  baseline_seconds: 2
  samplers: [cpu_power, gpu_power, ane_power]
thermal:
  abort_on_throttle: false
cells:
  - {{ id: c1, backend: passing, compute_unit: GPU, attention: NATIVE, precision: fp16, resolution: 512 }}
  - {{ id: c2, backend: passing, compute_unit: GPU, attention: NATIVE, precision: fp16, resolution: 512 }}
""",
        encoding="utf-8",
    )
    return path


def test_run_benchmark_writes_results_manifest_and_ledger(tmp_path):
    ws = Workspace.resolve(tmp_path)
    matrix = _write_matrix(tmp_path)

    records = run_benchmark(
        ws, matrix,
        adapters={"passing": PassingAdapter()},
        reporter=RunReporter(),
        power=False,
        use_plan=False,
    )

    assert {r.cell_id for r in records} == {"c1", "c2"}
    assert all(r.provenance_digest for r in records)
    assert (ws.results_data_dir / "results.jsonl").exists()
    assert (ws.results_data_dir / "environment.json").exists()
    assert ws.provenance_path.exists()
    assert (ws.shared_input_dir / "shared_input.npz").exists()  # generated on demand


def test_run_benchmark_captures_backend_output(tmp_path, capsys):
    ws = Workspace.resolve(tmp_path)
    matrix = _write_matrix(tmp_path)

    class NoisyAdapter(PassingAdapter):
        def prepare(self, cfg):
            print("LOADING pipeline components")
            print("a tracer warning", file=__import__("sys").stderr)
            return super().prepare(cfg)

    class CaptureReporter(RunReporter):
        capture_output = True

        def __init__(self):
            self.logs = []

        def log(self, message):
            self.logs.append(message)

    reporter = CaptureReporter()
    run_benchmark(ws, matrix, adapters={"passing": NoisyAdapter()}, reporter=reporter, power=False, use_plan=False)

    joined = "\n".join(reporter.logs)
    assert "LOADING pipeline components" in joined  # stdout captured into the reporter
    assert "a tracer warning" in joined  # stderr too
    assert "LOADING pipeline components" not in capsys.readouterr().out  # did not leak to the terminal


def test_clean_for_log_strips_ansi_and_cr():
    from sdbench.tui.run_cmd import _clean_for_log

    assert _clean_for_log("\x1b[31mhello\x1b[0m") == "hello"
    assert _clean_for_log("progress\r") == "progress"
    assert _clean_for_log("a\x1b[2Kb\x1b]0;title\x07c") == "abc"  # CSI + OSC removed


def test_capture_scrubs_ansi_so_it_cannot_corrupt_a_live(tmp_path):
    """Captured library output must not contain raw escape codes — they would re-enter
    the terminal through the Logs panel and corrupt the surrounding Live frame."""
    ws = Workspace.resolve(tmp_path)
    matrix = _write_matrix(tmp_path)

    class AnsiAdapter(PassingAdapter):
        def prepare(self, cfg):
            print("\x1b[33mWARNING:\x1b[0m colored stuff", flush=True)
            import os as _os
            _os.write(1, b"\x1b[1;32mGREEN native\x1b[0m\n")
            return super().prepare(cfg)

    class CaptureReporter(RunReporter):
        capture_output = True

        def __init__(self):
            self.logs = []

        def log(self, message):
            self.logs.append(message)

    reporter = CaptureReporter()
    run_benchmark(ws, matrix, adapters={"passing": AnsiAdapter()}, reporter=reporter, power=False, use_plan=False)
    joined = "\n".join(reporter.logs)
    assert "WARNING:" in joined and "GREEN native" in joined  # text preserved
    assert "\x1b" not in joined and "\r" not in joined  # escapes & CR scrubbed


def test_run_benchmark_captures_native_fd_writes(tmp_path):
    """Native (C-side) writes that bypass sys.stdout must still land in reporter.log."""
    ws = Workspace.resolve(tmp_path)
    matrix = _write_matrix(tmp_path)
    import os as _os

    class NativeAdapter(PassingAdapter):
        def prepare(self, cfg):
            _os.write(1, b"native fd-1 message\n")
            _os.write(2, b"native fd-2 error\n")
            return super().prepare(cfg)

    class CaptureReporter(RunReporter):
        capture_output = True

        def __init__(self):
            self.logs = []

        def log(self, message):
            self.logs.append(message)

    reporter = CaptureReporter()
    run_benchmark(ws, matrix, adapters={"passing": NativeAdapter()}, reporter=reporter, power=False, use_plan=False)

    joined = "\n".join(reporter.logs)
    assert "native fd-1 message" in joined
    assert "native fd-2 error" in joined


def test_run_benchmark_upserts_single_cell_without_clobbering(tmp_path):
    ws = Workspace.resolve(tmp_path)
    matrix = _write_matrix(tmp_path)
    run_benchmark(ws, matrix, adapters={"passing": PassingAdapter()}, reporter=RunReporter(), power=False, use_plan=False)

    # re-run only c1 -> c2 row must survive
    run_benchmark(ws, matrix, adapters={"passing": PassingAdapter()}, reporter=RunReporter(), power=False, use_plan=False, cell_ids=["c1"])

    rows = {r.cell_id for r in load_jsonl(ws.results_data_dir / "results.jsonl")}
    assert rows == {"c1", "c2"}
