"""Pre-run surfaces: disk footprint, power/sudo security disclosure, environment hints.

Disk math and footprint loading are pure and tested; the panels are presentation
only. Disk sizes are *measured* (populated by `measure-disk`), never estimated —
when a cell has no measured footprint yet it is reported as unknown, honestly.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from rich.panel import Panel

from sdbench import telemetry
from sdbench.thermal import check_thermal_state
from sdbench.tui.console import console, human_bytes

# An absolute loadavg ceiling is per-machine garbage: macOS counts threads in
# uninterruptible wait, not just runnable CPU, and macOS 26 idles a 10-core M2
# Pro around 2.5–3.0 once Spotlight/AddressBookManager/etc. are accounted for —
# the old "idles ≈ 0.5" assumption is dead. So we don't gate on an absolute
# number; we measure the host's *own* idle baseline once (with the operator's
# background already quiesced) and gate each pass on baseline + a margin. That
# catches *added* contamination (the intent of R6) instead of the box's floor.
#
# POWER_LOADAVG_MAX stays only as a fallback ceiling when no baseline was
# measured (e.g. a single ad-hoc run that skipped the session baseline step).
POWER_LOADAVG_MAX = 4.0

# How far above the measured idle baseline a pass may sit before its power
# numbers are treated as contaminated. Absolute headroom, not a ratio: a busy
# unrelated process adds roughly one unit of loadavg per pinned core, and we
# want to flag ~one extra busy core's worth of drift.
LOADAVG_MARGIN = 1.0


def measure_idle_loadavg(samples: int = 5, interval_s: float = 2.0, *, sleep=None) -> float | None:
    """Median 1-min loadavg over a short window — the host's own idle floor.

    Sampled once at session start, *after* the operator has quiesced their
    background, so it captures this machine/OS's baseline rather than assuming
    one. The median rejects a single transient spike. Returns None where
    ``getloadavg`` is unavailable (the gate then falls back to the absolute
    ceiling)."""
    import time as _time

    sleep = sleep or _time.sleep
    readings: list[float] = []
    for i in range(max(1, samples)):
        try:
            readings.append(os.getloadavg()[0])
        except (OSError, AttributeError):
            return None
        if i + 1 < samples:
            sleep(interval_s)
    readings.sort()
    return readings[len(readings) // 2]


def loadavg_ceiling(baseline: float | None, margin: float = LOADAVG_MARGIN) -> float:
    """Per-pass loadavg gate: idle baseline + margin, or the absolute fallback."""
    if baseline is None:
        return POWER_LOADAVG_MAX
    return baseline + margin


@dataclass(frozen=True)
class DiskReport:
    free_bytes: int
    known_required_bytes: int
    known_cells: dict[str, int]
    unknown_cells: list[str]

    @property
    def fully_known(self) -> bool:
        return not self.unknown_cells

    @property
    def fits(self) -> bool | None:
        if not self.fully_known:
            return None
        return self.free_bytes >= self.known_required_bytes


def load_footprint(path: str | Path) -> dict[str, int]:
    """Measured per-cell artifact sizes in bytes, or {} if not measured yet."""
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cells = raw.get("cells", {})
    return {str(key): int(value) for key, value in cells.items()}


def disk_report(free_bytes: int, footprint: dict[str, int], cell_ids: list[str]) -> DiskReport:
    known = {cid: footprint[cid] for cid in cell_ids if cid in footprint}
    unknown = [cid for cid in cell_ids if cid not in footprint]
    return DiskReport(
        free_bytes=free_bytes,
        known_required_bytes=sum(known.values()),
        known_cells=known,
        unknown_cells=unknown,
    )


def free_disk_bytes(path: str | Path) -> int:
    return shutil.disk_usage(Path(path)).free


def render_disk_report(report: DiskReport) -> None:
    lines = [f"Free space: [sdbench.size]{human_bytes(report.free_bytes)}[/]"]
    if report.known_cells:
        lines.append(f"Measured need (this selection): [sdbench.size]{human_bytes(report.known_required_bytes)}[/]")
    if report.unknown_cells:
        lines.append(
            f"[sdbench.warn]{len(report.unknown_cells)} cell(s) not yet measured[/] — "
            "footprint is populated after the first conversion (`sdbench measure-disk`)."
        )
    if report.fits is False:
        lines.append("[sdbench.danger]Not enough free space for the measured artifacts.[/]")
    elif report.fits is True:
        lines.append("[sdbench.ok]Enough free space for the measured artifacts.[/]")
    console.print(Panel("\n".join(lines), title="Disk", title_align="left", border_style="sdbench.dim"))


POWER_SECURITY_TEXT = (
    "Power measurement runs Apple's [bold]powermetrics[/], which needs root.\n"
    "I keep root to the minimum: [bold]only the sampler[/] runs under sudo — the\n"
    "benchmark harness itself stays unprivileged. You can audit exactly what is\n"
    "elevated in [bold]scripts/run.sh[/] and src/sdbench/tui/power_session.py before\n"
    "you type your password.\n\n"
    "Mitigations you can take: read those two files, run offline, and/or grant a\n"
    "time-boxed sudo session. If you would rather not, decline below — power\n"
    "metering is disabled and every other metric (latency, equivalence, size,\n"
    "conversion time) still runs."
)


def render_power_security() -> None:
    console.print(Panel(POWER_SECURITY_TEXT, title="Power needs sudo — here's why and how to stay safe", border_style="sdbench.warn"))


@dataclass(frozen=True)
class PowerEnvCheck:
    """Snapshot of the conditions that govern whether power numbers are usable.

    Each field carries the raw fact AND a boolean ``*_ok`` so the renderer can
    explain *why* a measurement was refused; the caller just consults ``ok``."""

    ac_powered: bool
    ac_ok: bool
    low_power_mode: bool
    low_power_ok: bool
    loadavg_1m: float | None
    loadavg_ok: bool
    issues: list[str]

    @property
    def ok(self) -> bool:
        # Only the un-waitable conditions hard-refuse here. A noisy loadavg is
        # *waited out* before each pass (see run_cmd._await_quiescent_host): the
        # 1-min EWMA carries our own tail between passes, so refusing on it would
        # false-positive on every pass after the first. loadavg_ok stays exposed
        # as an advisory/flag, but it does not gate ``ok``.
        return self.ac_ok and self.low_power_ok


def check_power_env(loadavg_max: float = POWER_LOADAVG_MAX) -> PowerEnvCheck:
    """Decide whether the host is in shape to deliver trustworthy power numbers.

    Battery-powered runs are refused: macOS reroutes thermal/clock budgets
    differently on battery and the per-engine W values stop being comparable to
    AC numbers. Low-power mode caps clocks. A noisy loadavg means the baseline
    captured at the start of the run is contaminated by an unrelated workload."""
    power_state = telemetry.collect_host_power_state()
    try:
        loadavg = os.getloadavg()[0]
    except (OSError, AttributeError):
        loadavg = None

    issues: list[str] = []
    if not power_state.ac_powered:
        issues.append("plug into AC power before measuring (battery throttles differently)")
    if power_state.low_power_mode:
        issues.append("low-power mode is on (System Settings → Battery)")
    if loadavg is not None and loadavg > loadavg_max:
        issues.append(
            f"loadavg_1m={loadavg:.2f} exceeds {loadavg_max:.1f} — close background apps "
            "(check Activity Monitor; macOS 26 ships an AddressBookManager regression)"
        )

    return PowerEnvCheck(
        ac_powered=power_state.ac_powered,
        ac_ok=power_state.ac_powered,
        low_power_mode=power_state.low_power_mode,
        low_power_ok=not power_state.low_power_mode,
        loadavg_1m=loadavg,
        loadavg_ok=loadavg is None or loadavg <= loadavg_max,
        issues=issues,
    )


def render_power_env(check: PowerEnvCheck) -> None:
    if check.ok:
        ac = "AC" if check.ac_powered else "battery"
        load = f"{check.loadavg_1m:.2f}" if check.loadavg_1m is not None else "?"
        console.print(f"[sdbench.ok]Power env: ok[/] ({ac}, loadavg_1m={load}).")
        return
    lines = [f"- {item}" for item in check.issues]
    console.print(
        Panel(
            "Power numbers will NOT be comparable in this environment:\n"
            + "\n".join(lines)
            + "\n\nFix the items above, then rerun. Override with --force-power if you "
            "really mean to record contaminated numbers (they will still be flagged "
            "in the manifest).",
            title="Power env check — refused",
            border_style="sdbench.danger",
        )
    )


def render_close_apps_reminder() -> None:
    console.print(
        Panel(
            "Close other heavy apps (browsers, builds, Docker, video) before running.\n"
            "Background load skews latency and pollutes the per-engine power baseline (R6).",
            title="Before you run",
            title_align="left",
            border_style="sdbench.dim",
        )
    )


def render_thermal_line() -> bool:
    """Print the current thermal state. Returns True if it is safe (not throttled)."""
    state = check_thermal_state()
    if state.throttled:
        console.print(f"[sdbench.danger]Thermal: throttled[/] ({state.detail}) — let the machine cool before timing (R5.6).")
        return False
    if state.source == "unavailable":
        console.print(f"[sdbench.warn]Thermal: unknown[/] ({state.detail}).")
        return True
    console.print(f"[sdbench.ok]Thermal: ok[/] ({state.detail}).")
    return True
