"""Interactive `config` flow: choose cells, power, verbosity; show preflight; save a run plan.

Keeps orchestration only — selection logic, gate evaluation, disk math, and the
run-plan model live in dedicated, tested modules. FULL SUITE is a separate,
deliberate menu choice, never the default (a full run must be conscious).
"""

import platform
import shutil

from sdbench.config import load_benchmark_config
from sdbench.tui.capabilities import detect_capabilities
from sdbench.tui.console import console
from sdbench.tui.preflight import (
    disk_report,
    free_disk_bytes,
    load_footprint,
    render_close_apps_reminder,
    render_disk_report,
    render_power_security,
    render_thermal_line,
)
from sdbench.tui.prompts import build_cell_rows, full_suite_ids, select_cells_interactive
from sdbench.tui.runplan import RunPlan, save_runplan
from sdbench.tui.workspace import Workspace


def power_available() -> tuple[bool, str]:
    if platform.system() != "Darwin":
        return False, "power sampling needs macOS powermetrics"
    if shutil.which("powermetrics") is None:
        return False, "powermetrics not found on PATH"
    return True, ""


def run_config(ws: Workspace, config_path) -> RunPlan | None:
    import questionary

    cfg = load_benchmark_config(config_path)
    caps = detect_capabilities()
    console.rule("[sdbench.title]sdbench · configure run[/]")
    console.print(f"Chip: [bold]{caps.chip}[/]")
    render_thermal_line()

    rows = build_cell_rows(cfg.cells, caps)
    locked = [r for r in rows if not r.selectable]
    if locked:
        console.print(f"[sdbench.dim]{len(locked)} cell(s) locked by capability gates; they will be recorded N/A.[/]")

    mode = questionary.select(
        "How do you want to pick cells?",
        choices=[
            questionary.Choice("Select cells individually (defaults pre-checked)", value="individual"),
            questionary.Choice("FULL SUITE — run every selectable cell", value="full"),
            questionary.Choice("Cancel", value="cancel"),
        ],
    ).ask()
    if mode in (None, "cancel"):
        console.print("[sdbench.dim]Cancelled. No run plan written.[/]")
        return None
    if mode == "full":
        cell_ids = full_suite_ids(rows)
    else:
        cell_ids = select_cells_interactive(rows)

    if not cell_ids:
        console.print("[sdbench.dim]No cells selected. No run plan written.[/]")
        return None

    available, reason = power_available()
    if not available:
        console.print(f"[sdbench.warn]Power metering disabled:[/] {reason}.")
        power_enabled = False
    else:
        render_power_security()
        power_enabled = bool(
            questionary.confirm("Enable power measurement (sudo for the sampler)?", default=False).ask()
        )

    verbosity = questionary.select(
        "Log verbosity during the run?",
        choices=["normal", "verbose", "quiet"],
        default="normal",
    ).ask() or "normal"

    run_conditions = questionary.text(
        "Run conditions note (optional, recorded in the manifest):",
        default="",
    ).ask() or ""

    report = disk_report(free_disk_bytes(ws.root), load_footprint(ws.disk_footprint_path), cell_ids)
    render_disk_report(report)
    render_close_apps_reminder()

    plan = RunPlan(
        cell_ids=cell_ids,
        power_enabled=power_enabled,
        verbosity=verbosity,
        run_conditions=run_conditions.strip(),
    )
    save_runplan(plan, ws.runplan_path)
    console.print(
        f"\n[sdbench.ok]Run plan saved[/] ({len(cell_ids)} cell(s), power {'on' if power_enabled else 'off'}). "
        "Start it with [bold]sdbench run[/]."
    )
    return plan
