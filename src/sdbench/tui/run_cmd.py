"""The delightful `run` orchestration: one call that does the whole right thing.

Resolves the run plan, invalidates results made stale by a changed checkpoint or
dependency set, ensures the shared input exists, gates on thermal throttling
(R5.6), runs the matrix while a reporter renders progress, measures power with
minimal root (only the sampler is elevated), stamps each record with the
provenance digest, upserts the results (never clobbers the full-matrix file), and
writes the environment manifest and provenance ledger.

All human-facing status goes through the reporter (``reporter.log``) so the same
flow drives either the line reporter (CLI) or the full-screen dashboard (guided
app). Heavy/privileged collaborators are injectable so the flow is testable with
fakes and without a Mac.
"""

import contextlib
import io
import os
import re
import threading
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from sdbench.config import load_benchmark_config
from sdbench.env import collect_environment_manifest, write_environment_manifest
from sdbench.inputs import generate_shared_input, load_shared_input, save_shared_input
from sdbench.orchestrator import run_matrix
from sdbench.power import apply_power_to_records, parse_powermetrics_plist
from sdbench.provenance import collect_fingerprint, invalidate_stale_results, record_run, sha256_file
from sdbench.results import upsert_jsonl, write_summary_tables
from sdbench.thermal import check_thermal_state
from sdbench.tui.power_session import PowerSession
from sdbench.tui.runplan import load_runplan
from sdbench.tui.workspace import Workspace

DEFAULT_SAMPLERS = ["cpu_power", "gpu_power", "ane_power"]


_NOISY_LOGGERS = ("coremltools", "transformers", "diffusers", "huggingface_hub", "torch", "mlx", "PIL", "urllib3")


def _silence_libraries() -> None:
    """Quiet third-party loggers and Python warnings.

    Some libraries print directly to fd (bypassing our capture) or hold their
    own stream references created at module-import time. Silencing them at the
    Python level (logger level + warnings filter) keeps the noise out of the
    full-screen UI at the source.
    """
    import logging
    import warnings

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")


def run_benchmark(
    ws: Workspace,
    config_path,
    *,
    cell_ids: list[str] | None = None,
    power: bool | None = None,
    verbosity: str | None = None,
    use_plan: bool = True,
    adapters=None,
    reporter=None,
    run_id: str | None = None,
    shared_input=None,
):
    _silence_libraries()
    cfg = load_benchmark_config(config_path)
    plan = load_runplan(ws.runplan_path) if (use_plan and ws.runplan_path.exists()) else None

    if cell_ids is None:
        cell_ids = plan.cell_ids if plan else [c.id for c in cfg.enabled_cells()]
    if power is None:
        power = plan.power_enabled if plan else False
    if verbosity is None:
        verbosity = plan.verbosity if plan else "normal"
    run_conditions = plan.run_conditions if plan else "default CLI run"

    if reporter is None:
        from sdbench.tui.run_view import SimpleReporter

        reporter = SimpleReporter(verbosity)

    cells = [cfg.select_cell_by_id(cid) for cid in cell_ids]
    if not cells:
        reporter.log("No cells selected; nothing to run.")
        return []
    cfg_run = replace(cfg, cells=cells)
    run_id = run_id or str(uuid4())

    checkpoint_sha = sha256_file(cfg.checkpoint) if Path(cfg.checkpoint).is_file() else None
    fingerprint = collect_fingerprint(ws, checkpoint_sha)
    removed = invalidate_stale_results(ws, fingerprint)
    if removed:
        reporter.log(f"Inputs changed since last run — invalidated {len(removed)} stale result file(s).")

    if shared_input is None:
        shared_input = _ensure_shared_input(ws, cfg_run, reporter)

    thermal = check_thermal_state()
    if thermal.throttled and cfg.thermal.abort_on_throttle:
        reporter.log(f"Aborting: thermal throttling detected ({thermal.detail}). Let the machine cool (R5.6).")
        return []
    if thermal.throttled:
        reporter.log(f"Thermal throttling detected ({thermal.detail}); cells will be flagged.")

    if adapters is None:
        from sdbench.backends.registry import build_default_adapters

        adapters = build_default_adapters(checkpoint_path=cfg.checkpoint)

    samplers = cfg.power.samplers or DEFAULT_SAMPLERS
    power_log = ws.results_raw_dir / f"{run_id}-powermetrics.plist"

    if power:
        from sdbench.tui.power_session import authorize_sudo

        reporter.log("Power measurement on — authorizing the powermetrics sampler (sudo)…")
        if not authorize_sudo():
            reporter.log("sudo not granted — continuing without power measurement.")
            power = False
    else:
        reporter.log("Power measurement off.")

    with PowerSession(log_path=power_log, interval_ms=cfg.power.interval_ms, samplers=samplers, enabled=bool(power)):
        with _capture_backend_output(reporter):
            records = run_matrix(
                cfg=cfg_run,
                shared_input=shared_input,
                adapters=adapters,
                run_id=run_id,
                results_dir=ws.results_dir,
                reporter=reporter,
            )

    if power:
        samples = parse_powermetrics_plist(power_log) if power_log.exists() else []
        if samples:
            records = apply_power_to_records(records, samples, cfg.power.baseline_seconds, cfg.iterations)
            reporter.log(f"Aligned {len(samples)} power samples to the timed windows.")
        else:
            reporter.log("Power was enabled but the sampler produced no samples; latency stands, power is N/A this run.")

    from sdbench.tui.convert_orchestrator import conversion_timings_by_cell

    timings = conversion_timings_by_cell(ws, cfg_run)
    records = [_with_conversion_timings(record, timings.get(record.cell_id)) for record in records]
    records = [replace(record, provenance_digest=fingerprint.digest) for record in records]

    upsert_jsonl(records, ws.results_data_dir / "results.jsonl")
    write_summary_tables(records, ws.results_tables_dir)
    write_environment_manifest(
        collect_environment_manifest(
            seed=cfg.seed,
            run_conditions=run_conditions,
            checkpoint_path=cfg.checkpoint,
            workspace=ws,
            provenance_digest=fingerprint.digest,
        ),
        ws.results_data_dir / "environment.json",
    )
    record_run(ws, fingerprint, run_id)
    return records


# Raw captured output is loaded with ANSI/control codes from torch/coremltools/tqdm.
# If those leak into the Logs panel they corrupt the surrounding terminal state
# (colour resets blank the panel borders, a stray ESC byte swallows the next char
# = the "shift by one" we saw). Strip them before handing to the sink.
_ANSI_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"        # CSI sequences (colours, cursor motion, …)
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC sequences (terminal titles, hyperlinks)
    r"|\x1b[@-_]"                      # short 2-byte escapes
)


def _clean_for_log(line: str) -> str:
    return _ANSI_RE.sub("", line).replace("\r", "")


class _LogForwarder(io.TextIOBase):
    """A write-only text stream that forwards complete lines to a sink (reporter.log)."""

    def __init__(self, sink) -> None:
        super().__init__()
        self._sink = sink
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            cleaned = _clean_for_log(line)
            if cleaned.strip():
                self._sink(cleaned)
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            self._sink(_clean_for_log(self._buf))
        self._buf = ""


@contextlib.contextmanager
def _capture_backend_output(reporter):
    """Funnel backend stdout/stderr into reporter.log for the full-screen dashboard.

    Two layers, because each catches what the other misses:

    * **fd-level** (`os.dup2` of fds 1/2 → a pipe; background thread forwards
      lines to ``reporter.log``). This catches *native* writes from torch /
      coremltools / mlx that bypass Python's ``sys.stdout``. The Live's console
      is already pinned by ``screen.live_screen()`` to a saved duplicate fd of
      the real terminal, so its frame keeps drawing while fds 1/2 are diverted.
    * **Python-level** (``redirect_stdout``/``stderr`` to a line forwarder).
      This catches prints intercepted by Python wrappers (notably pytest's
      ``capsys``) before they ever reach fd 1.
    """
    if not getattr(reporter, "capture_output", False):
        yield
        return

    forwarder = _LogForwarder(reporter.log)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _reader() -> None:
        try:
            with os.fdopen(read_fd, "r", buffering=1, errors="replace") as stream:
                for line in stream:
                    cleaned = _clean_for_log(line.rstrip("\n"))
                    if cleaned.strip():
                        reporter.log(cleaned)
        except (ValueError, OSError):
            pass

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    try:
        with contextlib.redirect_stdout(forwarder), contextlib.redirect_stderr(forwarder):
            yield
    finally:
        forwarder.flush()
        # Restore the original fds (this drops the last refs to the pipe write end,
        # so the reader sees EOF and exits).
        try:
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
        finally:
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
        thread.join(timeout=2)


def _with_conversion_timings(record, timings):
    if not timings:
        return record
    return replace(
        record,
        graph_capture_s=timings.get("graph_capture_s"),
        convert_s=timings.get("convert_s"),
        first_load_compile_s=timings.get("first_load_compile_s"),
    )


def _ensure_shared_input(ws: Workspace, cfg_run, reporter):
    path = ws.shared_input_dir / "shared_input.npz"
    if not path.exists():
        resolutions = {cell.resolution for cell in cfg_run.cells}
        if len(resolutions) != 1:
            raise ValueError("Shared input generation requires one resolution across selected cells")
        shared = generate_shared_input(seed=cfg_run.seed, resolution=resolutions.pop())
        save_shared_input(shared, path)
        reporter.log(f"Generated shared input → {path}")
    return load_shared_input(path)
