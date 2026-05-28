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
from sdbench.tui.prompts import build_cell_rows
from sdbench.tui.runplan import RunPlan, save_runplan

_VERBOSITY = ["normal", "verbose", "quiet"]


class MatrixModel:
    def __init__(self, rows, power_default: bool = False) -> None:
        self.rows = rows
        self.index = 0
        self.selected = {r.cell.id for r in rows if r.default_selected}
        self.power = power_default
        self.verbosity = "normal"

    def move(self, delta: int) -> None:
        if self.rows:
            self.index = (self.index + delta) % len(self.rows)

    def toggle(self) -> None:
        row = self.rows[self.index]
        if not row.selectable:
            return
        cid = row.cell.id
        self.selected.discard(cid) if cid in self.selected else self.selected.add(cid)

    def select_all(self) -> None:
        self.selected = {r.cell.id for r in self.rows if r.selectable}

    def clear(self) -> None:
        self.selected.clear()

    def cycle_verbosity(self) -> None:
        self.verbosity = _VERBOSITY[(_VERBOSITY.index(self.verbosity) + 1) % len(_VERBOSITY)]

    def chosen_ids(self) -> list[str]:
        return [r.cell.id for r in self.rows if r.cell.id in self.selected]


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


def _render(ws, model: MatrixModel, power_ok: bool, power_reason: str, cfg):
    settings = Text()
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
        screen.footer("↑/↓ move · space toggle · f FULL SUITE · a all · n none · p power · v verbosity · s save · q cancel"),
    )


def config_view(live, ws, config_path) -> RunPlan | None:
    cfg = load_benchmark_config(config_path)
    rows = build_cell_rows(cfg.cells, detect_capabilities())
    model = MatrixModel(rows)
    power_ok, power_reason = power_available()

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
        elif key == "f":
            model.select_all()
        elif key == "a":
            model.select_all()
        elif key == "n":
            model.clear()
        elif key == "p" and power_ok:
            model.power = not model.power
        elif key == "v":
            model.cycle_verbosity()
        elif key == "s" and model.chosen_ids():
            plan = RunPlan(
                cell_ids=model.chosen_ids(),
                power_enabled=model.power and power_ok,
                verbosity=model.verbosity,
                run_conditions="",
            )
            save_runplan(plan, ws.runplan_path)
            return plan


def run_config_screen(ws, config_path) -> RunPlan | None:
    """Standalone full-screen config editor (for the `config` subcommand)."""
    with screen.live_screen() as live:
        return config_view(live, ws, config_path)
