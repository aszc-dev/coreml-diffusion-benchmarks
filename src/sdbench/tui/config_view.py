"""Full-screen run configuration: the matrix as a laid-out grid, not a checkbox list.

Arrow keys move a cursor over the matrix table; space toggles a cell; locked
cells (capability gate unmet) cannot be selected. Power and verbosity are
toggles. FULL SUITE is its own key ('f'), never the default — a full run stays a
conscious choice. Saving writes a RunPlan for `run`.

The MatrixModel is pure and tested; the loop only maps keys to it and renders.
"""

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sdbench.config import load_benchmark_config
from sdbench.tui import screen
from sdbench.tui.capabilities import detect_capabilities
from sdbench.tui.config_cmd import power_available
from sdbench.tui.prompts import build_cell_rows, full_suite_ids
from sdbench.tui.runplan import (
    RunPlan,
    fast_test_preset,
    load_runplan,
    publication_preset,
    save_runplan,
)

_VERBOSITY = ["normal", "verbose", "quiet"]


class MatrixModel:
    def __init__(
        self,
        rows,
        power_default: bool = False,
        *,
        initial_selected: set[str] | None = None,
        initial_verbosity: str | None = None,
        initial_mode: str = "publication",
        initial_repeats: int = 5,
        initial_iterations: int | None = None,
    ) -> None:
        self.rows = rows
        self.index = 0
        # Saved selection (if any) wins over the default — opening the editor
        # twice in a row must show what the user last saved, not a fresh
        # default set. Locked rows are filtered so a capability that vanished
        # since last save doesn't re-appear as a phantom selection.
        if initial_selected is not None:
            selectable = {r.cell.id for r in rows if r.selectable}
            self.selected = {cid for cid in initial_selected if cid in selectable}
        else:
            self.selected = {r.cell.id for r in rows if r.default_selected}
        self.power = power_default
        self.verbosity = initial_verbosity if initial_verbosity in _VERBOSITY else "normal"
        self.mode = initial_mode if initial_mode in ("publication", "fast", "custom") else "publication"
        self.repeats = max(1, int(initial_repeats))
        self.cooldown_s = 30.0 if self.repeats > 1 else 0.0
        self.iterations = initial_iterations  # None = use matrix.yaml default

    def move(self, delta: int) -> None:
        if self.rows:
            self.index = (self.index + delta) % len(self.rows)

    def toggle(self) -> None:
        row = self.rows[self.index]
        if not row.selectable:
            return
        cid = row.cell.id
        self.selected.discard(cid) if cid in self.selected else self.selected.add(cid)
        # Manual cell editing means the user is no longer on a clean preset.
        self.mode = "custom"

    def select_all(self) -> None:
        self.selected = {r.cell.id for r in self.rows if r.selectable}
        self.mode = "custom"

    def clear(self) -> None:
        self.selected.clear()
        self.mode = "custom"

    def cycle_verbosity(self) -> None:
        self.verbosity = _VERBOSITY[(_VERBOSITY.index(self.verbosity) + 1) % len(_VERBOSITY)]
        self.mode = "custom"

    def apply_publication_preset(self, power_ok: bool) -> None:
        """Snap to the publication preset: every enabled+selectable cell,
        multi-run aggregate, power on if the host supports it. Reads exactly
        like :func:`publication_preset` — the TUI is a thin editor over the
        same shape the headless flow produces."""
        self.selected = set(full_suite_ids(self.rows))
        self.power = power_ok
        self.verbosity = "normal"
        self.repeats = 5
        self.cooldown_s = 30.0
        self.iterations = None
        self.mode = "publication"

    def apply_fast_test_preset(self) -> None:
        """Snap to the fast-test preset: one cell, one pass, R5.3 floor."""
        head = next((r.cell.id for r in self.rows if r.selectable and r.cell.enabled), None)
        if head is None:
            # No enabled+selectable cell — fall back to any selectable so the
            # preset still lands somewhere actionable.
            head = next((r.cell.id for r in self.rows if r.selectable), None)
        self.selected = {head} if head else set()
        self.power = False
        self.verbosity = "quiet"
        self.repeats = 1
        self.cooldown_s = 0.0
        self.iterations = 10
        self.mode = "fast"

    def chosen_ids(self) -> list[str]:
        return [r.cell.id for r in self.rows if r.cell.id in self.selected]

    def to_plan(self, *, power_ok: bool, run_conditions: str = "") -> RunPlan:
        """Materialise the current model state as a :class:`RunPlan`.

        Power is gated on host capability separately so the saved plan never
        promises measurement on a machine that can't deliver it."""
        return RunPlan(
            cell_ids=self.chosen_ids(),
            power_enabled=bool(self.power and power_ok),
            verbosity=self.verbosity,
            run_conditions=run_conditions,
            repeats=self.repeats,
            cooldown_s=self.cooldown_s,
            iterations=self.iterations,
            mode=self.mode,
        )


def _matrix_table(model: MatrixModel) -> Table:
    table = Table(expand=True, header_style="sdbench.dim", border_style="sdbench.dim")
    table.add_column(" ", width=2)
    table.add_column("Sel", width=3)
    table.add_column("Cell")
    table.add_column("Backend")
    table.add_column("Prec")
    table.add_column("State")
    for i, row in enumerate(model.rows):
        cursor = "▶" if i == model.index else " "
        if not row.selectable:
            box = "[sdbench.dim]·[/]"
            state = f"[sdbench.warn]{row.status_label}[/]"
        else:
            box = "[sdbench.ok]x[/]" if row.cell.id in model.selected else " "
            state = f"[sdbench.dim]{row.status_label}[/]"
        style = "reverse" if i == model.index else None
        table.add_row(cursor, box, row.cell.label or row.cell.id, row.cell.backend, row.cell.precision, state, style=style)
    return table


_MODE_STYLE = {
    "publication": ("sdbench.ok", "publication (multi-run aggregate, default)"),
    "fast": ("sdbench.warn", "fast test (1 pass, 10 iters, no power)"),
    "custom": ("sdbench.title", "custom"),
}


def _render(ws, model: MatrixModel, power_ok: bool, power_reason: str, cfg):
    settings = Text()
    style, label = _MODE_STYLE.get(model.mode, _MODE_STYLE["custom"])
    settings.append("mode ", style="sdbench.dim")
    settings.append(f"{label}\n", style=style)

    settings.append("repeats ", style="sdbench.dim")
    settings.append(str(model.repeats), style="sdbench.title")
    if model.repeats > 1:
        settings.append(
            f" (aggregate median + p10/p90, cooldown {model.cooldown_s:.0f}s)   ",
            style="sdbench.dim",
        )
    else:
        settings.append("  (single pass, no aggregate)   ", style="sdbench.dim")

    settings.append("iters ", style="sdbench.dim")
    if model.iterations:
        settings.append(f"{model.iterations} (override)   ", style="sdbench.warn")
    else:
        settings.append(f"{cfg.iterations} (matrix.yaml)   ", style="sdbench.title")

    settings.append("\n")
    if power_ok:
        settings.append("power ", style="sdbench.dim")
        settings.append("ON " if model.power else "off ", style="sdbench.ok" if model.power else "sdbench.warn")
        settings.append("(only the sampler runs as root)   ", style="sdbench.dim")
    else:
        settings.append(f"power unavailable ({power_reason})   ", style="sdbench.warn")
    settings.append("verbosity ", style="sdbench.dim")
    settings.append(model.verbosity, style="sdbench.title")
    settings.append(f"   ·   {len(model.chosen_ids())} cell(s) selected", style="sdbench.dim")

    body = Table.grid(expand=True)
    body.add_row(_matrix_table(model))
    body.add_row(Panel(settings, border_style="sdbench.dim", padding=(0, 1)))

    from sdbench.tui.app import assess_state  # local import avoids a cycle at import time

    state = assess_state(ws, cfg)
    return screen.frame(
        screen.header("sdbench · configure run", screen.state_text(state), screen.usage_text(ws)),
        Panel(body, title="Matrix", border_style="sdbench.title"),
        screen.footer(
            "↑/↓ move · space toggle · P publication · T fast-test · +/- repeats "
            "· a all · n none · p power · v verbosity · s save · q cancel"
        ),
    )


def config_view(live, ws, config_path) -> RunPlan | None:
    cfg = load_benchmark_config(config_path)
    rows = build_cell_rows(cfg.cells, detect_capabilities())
    # Re-hydrate from the last saved run plan so re-opening the editor shows
    # the current configuration, not the fresh default — otherwise every visit
    # forces the user to redo their selection just to inspect it.
    saved: RunPlan | None = None
    if ws.runplan_path.exists():
        try:
            saved = load_runplan(ws.runplan_path)
        except (OSError, ValueError, KeyError):
            saved = None  # corrupt/old schema → fall back to defaults
    power_ok, power_reason = power_available()
    model = MatrixModel(
        rows,
        power_default=bool(saved.power_enabled) and power_ok if saved else power_ok,
        initial_selected=set(saved.cell_ids) if saved else None,
        initial_verbosity=saved.verbosity if saved else None,
        initial_mode=saved.mode if saved else "publication",
        initial_repeats=saved.repeats if saved else 5,
        initial_iterations=saved.iterations if saved else None,
    )
    # First visit: snap to the publication preset so the user lands on a
    # ready-to-save plan that reflects what the tool was built for (the
    # multi-run aggregate). Editing anything flips ``mode`` to ``custom``.
    if saved is None:
        model.apply_publication_preset(power_ok)

    while True:
        live.update(_render(ws, model, power_ok, power_reason, cfg))
        live.refresh()
        key = screen.read_key()
        if key in (screen.ESC, "q"):
            return None
        if key == screen.UP:
            model.move(-1)
        elif key == screen.DOWN:
            model.move(1)
        elif key == screen.SPACE:
            model.toggle()
        elif key in ("P",):
            model.apply_publication_preset(power_ok)
        elif key in ("T",):
            model.apply_fast_test_preset()
        elif key in ("+", "="):
            # ``=`` is the unshifted ``+`` on US layouts; accept both so the
            # user doesn't have to chord.
            model.repeats = min(model.repeats + 1, 20)
            if model.repeats > 1 and model.cooldown_s == 0.0:
                model.cooldown_s = 30.0
            model.mode = "custom"
        elif key == "-":
            model.repeats = max(model.repeats - 1, 1)
            if model.repeats == 1:
                model.cooldown_s = 0.0
            model.mode = "custom"
        elif key in ("f", "a"):
            model.select_all()
        elif key == "n":
            model.clear()
        elif key == "p" and power_ok:
            model.power = not model.power
            model.mode = "custom"
        elif key == "v":
            model.cycle_verbosity()
        elif key == "s" and model.chosen_ids():
            plan = model.to_plan(power_ok=power_ok)
            save_runplan(plan, ws.runplan_path)
            return plan


def run_config_screen(ws, config_path) -> RunPlan | None:
    """Standalone full-screen config editor (for the `config` subcommand)."""
    with screen.live_screen() as live:
        return config_view(live, ws, config_path)
