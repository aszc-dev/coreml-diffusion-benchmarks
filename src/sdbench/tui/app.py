"""Guided full-screen front door (Rich Live on the alternate screen).

`sdbench` with no subcommand lands here: a header with workspace state and a
live readout of how much disk our own files use, a four-action menu (Convert /
Configure / Run / Clean up), and views that redraw in place — same data as the
subcommands, a flashier UX. The Live is suspended around steps that need the
real terminal (sudo prompt, model download, toolchain output).
"""

from dataclasses import dataclass
from pathlib import Path

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sdbench.config import load_benchmark_config
from sdbench.tui import screen
from sdbench.tui.console import console, human_bytes
from sdbench.tui.workspace import Workspace

MENU = [
    ("convert", "Convert artifacts"),
    ("config", "Configure run"),
    ("run", "Run benchmark"),
    ("report", "Build submission report"),
    ("cleanup", "Clean up"),
    ("quit", "Quit"),
]


@dataclass(frozen=True)
class WorkspaceState:
    checkpoint: Path | None
    has_runplan: bool
    artifacts_present: int
    artifacts_total: int
    has_results: bool
    has_report: bool

    @property
    def checkpoint_present(self) -> bool:
        return self.checkpoint is not None


def assess_state(ws: Workspace, cfg) -> WorkspaceState:
    from sdbench.tui.convert_orchestrator import plan_conversions
    from sdbench.tui.download import DEFAULT_CHECKPOINT

    checkpoint = None
    if Path(cfg.checkpoint).is_file():
        checkpoint = Path(cfg.checkpoint)
    else:
        cached = ws.cache_dir / DEFAULT_CHECKPOINT.filename
        if cached.is_file():
            checkpoint = cached

    builds = plan_conversions(ws, cfg)
    present = sum(1 for build in builds if build.expected_artifact.exists())
    reports_dir = ws.results_dir / "reports"
    has_report = reports_dir.is_dir() and any(reports_dir.glob("*.zip"))
    return WorkspaceState(
        checkpoint=checkpoint,
        has_runplan=ws.runplan_path.exists(),
        artifacts_present=present,
        artifacts_total=len(builds),
        has_results=(ws.results_data_dir / "results.jsonl").exists(),
        has_report=has_report,
    )


class Menu:
    def __init__(self, items) -> None:
        self.items = items
        self.index = 0

    def move(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.items)

    @property
    def key(self) -> str:
        return self.items[self.index][0]


def guided_main(ws: Workspace, config_path) -> None:
    menu = Menu(MENU)
    with screen.live_screen() as live:
        while True:
            cfg = load_benchmark_config(config_path)
            live.update(_menu_frame(ws, cfg, menu))
            live.refresh()
            key = screen.read_key()
            if key in (screen.ESC, "q"):
                return
            if key == screen.UP:
                menu.move(-1)
            elif key == screen.DOWN:
                menu.move(1)
            elif key == screen.ENTER:
                if menu.key == "quit":
                    return
                _dispatch(menu.key, live, ws, config_path)


def _menu_frame(ws: Workspace, cfg, menu: Menu):
    state = assess_state(ws, cfg)
    body = Text()
    for i, (_, label) in enumerate(menu.items):
        if i == menu.index:
            body.append(f"  ▶ {label}\n", style="sdbench.title")
        else:
            body.append(f"    {label}\n", style="sdbench.dim")
    body.append("\n  Convert offers to download the model if it is missing.\n", style="sdbench.dim")
    body.append("  Run will send you to configure / convert first if needed.\n", style="sdbench.dim")
    return screen.frame(
        screen.header("cdbench · SD 1.5 UNet benchmark", screen.state_text(state), screen.usage_text(ws)),
        Panel(body, title="Menu", border_style="sdbench.title"),
        screen.footer("↑/↓ move · enter select · q quit"),
    )


def _dispatch(action: str, live, ws: Workspace, config_path) -> None:
    if action == "config":
        from sdbench.tui.config_view import config_view

        config_view(live, ws, config_path)
    elif action == "convert":
        _convert_flow(live, ws, config_path)
    elif action == "run":
        _run_flow(live, ws, config_path)
    elif action == "report":
        _report_flow(live, ws, config_path)
    elif action == "cleanup":
        _cleanup_flow(live, ws, config_path)


# ----- shared full-screen helpers -----

def _header(ws: Workspace, cfg, title: str):
    return screen.header(title, screen.state_text(assess_state(ws, cfg)), screen.usage_text(ws))


def _notice(live, ws, config_path, title: str, message: str) -> None:
    cfg = load_benchmark_config(config_path)
    live.update(screen.frame(_header(ws, cfg, title), Panel(Text(message), border_style="sdbench.dim"), screen.footer("press any key")))
    live.refresh()
    screen.read_key()


def _confirm(live, ws, config_path, question: str) -> bool:
    cfg = load_benchmark_config(config_path)
    panel = Panel(Text(f"{question}\n\n[y] yes     [n] no", style="sdbench.title"), title="Confirm", border_style="sdbench.warn")
    live.update(screen.frame(_header(ws, cfg, "Confirm"), panel, screen.footer("y / n")))
    live.refresh()
    while True:
        key = screen.read_key()
        if key in ("y", screen.ENTER):
            return True
        if key in ("n", "q", screen.ESC):
            return False


def _pause() -> None:
    try:
        input("\nPress Enter to return to the menu… ")
    except (EOFError, KeyboardInterrupt):
        pass


# ----- flows -----

def _convert_flow(live, ws: Workspace, config_path) -> None:
    from sdbench.provenance import sha256_file
    from sdbench.tui.convert_orchestrator import convert_all, plan_conversions
    from sdbench.tui.dashboard import ConvertDashboard

    cfg = load_benchmark_config(config_path)
    checkpoint = _resolve_checkpoint(live, ws, config_path)
    if checkpoint is None:
        return

    dashboard = ConvertDashboard(live, ws, plan_conversions(ws, cfg))
    dashboard.on_line("hashing checkpoint…")
    try:
        convert_all(
            ws, cfg, checkpoint, sha256_file(checkpoint),
            on_build=dashboard.on_build, on_line=dashboard.on_line,
            on_skip=dashboard.on_skip, on_done=dashboard.on_done,
        )
    except Exception as exc:  # toolchain failures shouldn't kill the menu
        dashboard.on_error(str(exc))
    dashboard.finish()
    screen.invalidate_usage(ws)
    screen.read_key()


def _resolve_checkpoint(live, ws: Workspace, config_path) -> Path | None:
    from sdbench.tui.download import DEFAULT_CHECKPOINT, resolve_checkpoint

    cfg = load_benchmark_config(config_path)
    if Path(cfg.checkpoint).is_file():
        return Path(cfg.checkpoint)
    cached = ws.cache_dir / DEFAULT_CHECKPOINT.filename
    if cached.is_file():
        return cached
    if not _confirm(live, ws, config_path, f"Checkpoint not found. Download {DEFAULT_CHECKPOINT.filename} from the official HF repo and verify its SHA?"):
        return None
    live.stop()
    result: Path | None = None
    try:
        result = resolve_checkpoint(ws, explicit=None, auto_download=True)
    except Exception as exc:
        console.print(f"[sdbench.danger]Download failed:[/] {exc}")
    _pause()
    live.start()
    return result if (result and Path(result).is_file()) else None


def _run_flow(live, ws: Workspace, config_path) -> None:
    from sdbench.tui.config_view import config_view
    from sdbench.tui.convert_orchestrator import plan_conversions
    from sdbench.tui.dashboard import DashboardReporter
    from sdbench.tui.run_cmd import run_benchmark
    from sdbench.tui.runplan import load_runplan

    if not ws.runplan_path.exists():
        if not _confirm(live, ws, config_path, "No run plan yet. Configure one now?"):
            return
        if config_view(live, ws, config_path) is None:
            return

    cfg = load_benchmark_config(config_path)
    missing = [b for b in plan_conversions(ws, cfg) if not b.expected_artifact.exists()]
    if missing and _confirm(live, ws, config_path, f"{len(missing)} CoreML artifact(s) not converted yet. Convert now?"):
        _convert_flow(live, ws, config_path)

    plan = load_runplan(ws.runplan_path)
    # Authorize sudo with the Live suspended so the password prompt is visible.
    if plan.power_enabled:
        from sdbench.tui.power_session import authorize_sudo

        live.stop()
        console.print("[sdbench.title]Power measurement on[/] — authorizing the powermetrics sampler (sudo)…")
        authorize_sudo()
        live.start()

    dashboard = DashboardReporter(live, ws, cell_ids=plan.cell_ids)
    records = run_benchmark(ws, config_path, reporter=dashboard)
    dashboard.show_summary(records)
    screen.read_key()
    # Offer the report flow inline so the contributor lands the bundle in one
    # session — same workflow as the CLI, but guided.
    if records and _confirm(
        live, ws, config_path,
        "Build a submission bundle now (manifest + JSONL + plist + tables, R11.14)?",
    ):
        _report_flow(live, ws, config_path)


def _report_flow(live, ws: Workspace, config_path) -> None:
    """Build a contributor submission bundle from the latest run + validate it.

    Same shape as ``cdbench report`` + ``cdbench validate-report`` but driven
    interactively from the guided flow: confirm anonymization, prompt for the
    salt (with explanation), build the bundle, gate it through the validator,
    and surface the resulting zip path plus next-step instructions.
    """
    import json

    from sdbench.report import build_report, validate_report

    manifest_path = ws.results_data_dir / "environment.json"
    if not manifest_path.exists():
        _notice(
            live, ws, config_path, "Build submission report",
            "No environment.json yet — run the benchmark first (Run → benchmark).\n"
            "The bundle is built from the latest run's manifest + results.",
        )
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _notice(live, ws, config_path, "Build submission report", f"Could not read manifest: {exc}")
        return

    run_id = manifest.get("run_id") or "latest"
    chip = (manifest.get("hardware") or {}).get("chip_brand") or "(unknown chip)"
    build = (manifest.get("os") or {}).get("build_version") or "(unknown build)"
    digest = manifest.get("provenance_digest") or "(unknown)"
    cells_run = manifest.get("cells_run") or []

    summary = Text()
    summary.append(f"  Run ID         {run_id}\n")
    summary.append(f"  Chip           {chip}\n")
    summary.append(f"  macOS build    {build}\n")
    summary.append(f"  Provenance     {digest[:12] if digest != '(unknown)' else digest}\n")
    summary.append(f"  Cells run      {len(cells_run)}\n")
    panel = Panel(summary, title="Source run", border_style="sdbench.title")
    live.update(screen.frame(
        _header(ws, load_benchmark_config(config_path), "Build submission report"),
        panel,
        screen.footer("press any key to continue"),
    ))
    live.refresh()
    screen.read_key()

    # Anonymization is opt-in. If the user is going to publish or share with a
    # maintainer they don't fully trust, this strips filesystem-local PII and
    # re-hashes the host id. R11.14 forbids stripping anything reproducibility
    # actually needs (seed, SHAs, package versions stay).
    anonymize = _confirm(
        live, ws, config_path,
        "Anonymize the bundle? Strips checkpoint_path / repo upstream / dirty file list,\n"
        "and re-hashes host_id_hash with your salt. Seed, SHAs, package versions stay.",
    )
    salt: str | None = None
    if anonymize:
        live.stop()
        try:
            import questionary

            console.print(
                "[sdbench.title]Anonymization salt[/] — used to re-hash your host id.\n"
                "Reuse the same salt across runs so your own datapoints still dedup.\n"
                "Treat it like a passphrase: don't share it with the maintainer."
            )
            salt = questionary.text(
                "Salt:",
                validate=lambda v: bool(v.strip()) or "Salt cannot be empty.",
            ).ask()
        finally:
            live.start()
        if not salt:
            _notice(live, ws, config_path, "Build submission report", "No salt entered — bundle not built.")
            return

    # Build + validate, both with Live paused around the questionary path; the
    # bundle build is fast (copy + zip), so a panel + post-hoc summary is enough,
    # no progress bar needed.
    live.update(screen.frame(
        _header(ws, load_benchmark_config(config_path), "Build submission report"),
        Panel(Text("  Building bundle…\n"), border_style="sdbench.dim"),
        screen.footer(""),
    ))
    live.refresh()
    try:
        bundle_path = build_report(
            ws,
            run_id=manifest.get("run_id"),
            zip_bundle=True,
            anonymize=anonymize,
            salt=salt,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        _notice(live, ws, config_path, "Build submission report", f"Build failed: {exc}")
        return

    bundle_dir = (ws.results_dir / "reports" / (manifest.get("run_id") or "latest"))
    verdict = validate_report(bundle_dir)

    result = Text()
    result.append("  Bundle:    ")
    result.append(f"{bundle_path}\n", style="sdbench.ok")
    result.append("  Directory: ")
    result.append(f"{bundle_dir}\n\n")
    if verdict.ok:
        result.append("  validate-report: ", style="sdbench.dim")
        result.append("OK\n\n", style="sdbench.ok")
    else:
        result.append("  validate-report: ", style="sdbench.dim")
        result.append("FAILED\n", style="sdbench.danger")
        for issue in verdict.issues:
            result.append(f"    - {issue}\n", style="sdbench.danger")
        result.append("\n")
    result.append("  Schema version: ", style="sdbench.dim")
    result.append(f"{verdict.schema_version} (supported: {verdict.supported_schema})\n")
    result.append("\n  Next: attach the .zip to a GitHub Discussion in this repo.\n", style="sdbench.title")
    if anonymize:
        result.append("  Keep your salt secret — you'll need it to dedup your future runs.\n", style="sdbench.dim")

    border = "sdbench.ok" if verdict.ok else "sdbench.warn"
    live.update(screen.frame(
        _header(ws, load_benchmark_config(config_path), "Build submission report"),
        Panel(result, title="Bundle ready", border_style=border),
        screen.footer("press any key"),
    ))
    live.refresh()
    screen.read_key()


def _cleanup_flow(live, ws: Workspace, config_path) -> None:
    from sdbench.tui.cleanup import delete_target, discover_targets

    targets = discover_targets(ws)
    if not targets:
        _notice(live, ws, config_path, "Clean up", "Nothing to clean — the workspace is already tidy.")
        return

    selected: set[str] = set()
    index = 0
    cfg = load_benchmark_config(config_path)
    while True:
        live.update(screen.frame(_header(ws, cfg, "Clean up"), _cleanup_table(targets, selected, index),
                                 screen.footer("↑/↓ move · space mark · d delete marked · q cancel")))
        live.refresh()
        key = screen.read_key()
        if key in ("q", screen.ESC):
            return
        if key == screen.UP:
            index = (index - 1) % len(targets)
        elif key == screen.DOWN:
            index = (index + 1) % len(targets)
        elif key == screen.SPACE:
            tkey = targets[index].key
            selected.discard(tkey) if tkey in selected else selected.add(tkey)
        elif key == "d" and selected:
            chosen = [t for t in targets if t.key in selected]
            total = sum(t.size_bytes for t in chosen)
            if _confirm(live, ws, config_path, f"Delete {len(chosen)} group(s), freeing {human_bytes(total)}? This cannot be undone."):
                for target in chosen:
                    delete_target(target)
                screen.invalidate_usage(ws)
                _notice(live, ws, config_path, "Clean up", f"Freed {human_bytes(total)}.")
                return


def _cleanup_table(targets, selected, index):
    table = Table(expand=True, header_style="sdbench.dim", border_style="sdbench.dim")
    table.add_column(" ", width=2)
    table.add_column("Mark", width=4)
    table.add_column("Target")
    table.add_column("Size", justify="right", style="sdbench.size")
    table.add_column("Items", justify="right")
    for i, target in enumerate(targets):
        cursor = "▶" if i == index else " "
        box = "[sdbench.ok]x[/]" if target.key in selected else " "
        style = "reverse" if i == index else None
        table.add_row(cursor, box, target.label, human_bytes(target.size_bytes), str(len(target.paths)), style=style)
    return Panel(table, title="Reclaimable", border_style="sdbench.title")
