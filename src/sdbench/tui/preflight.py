"""Pre-run surfaces: disk footprint, power/sudo security disclosure, environment hints.

Disk math and footprint loading are pure and tested; the panels are presentation
only. Disk sizes are *measured* (populated by `measure-disk`), never estimated —
when a cell has no measured footprint yet it is reported as unknown, honestly.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from rich.panel import Panel

from sdbench.thermal import check_thermal_state
from sdbench.tui.console import console, human_bytes


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
