"""Cell-selection rows and the interactive checkbox picker for `config`.

Pure row construction (cell + gate + default-selected) is separated from the
questionary picker so selection logic is testable without a TTY. Cells the host
cannot run (unmet/unknown capability gate) are shown locked with a reason; they
are never silently dropped — at run time they are recorded N/A (R8.4).
"""

from dataclasses import dataclass

from sdbench.config import CellConfig
from sdbench.tui.capabilities import Capabilities, GateStatus, evaluate_gate


@dataclass(frozen=True)
class CellRow:
    cell: CellConfig
    gate: GateStatus
    default_selected: bool

    @property
    def selectable(self) -> bool:
        return self.gate.selectable

    @property
    def status_label(self) -> str:
        if self.selectable:
            return "ready" if self.cell.enabled else "optional"
        return f"locked — {self.gate.detail}"


def build_cell_rows(cells: list[CellConfig], caps: Capabilities) -> list[CellRow]:
    rows: list[CellRow] = []
    for cell in cells:
        gate = evaluate_gate(caps, cell.requires)
        rows.append(CellRow(cell=cell, gate=gate, default_selected=cell.enabled and gate.selectable))
    return rows


def full_suite_ids(rows: list[CellRow]) -> list[str]:
    """Every selectable cell — the deliberate FULL SUITE choice."""
    return [row.cell.id for row in rows if row.selectable]


def default_ids(rows: list[CellRow]) -> list[str]:
    return [row.cell.id for row in rows if row.default_selected]


def select_cells_interactive(rows: list[CellRow]) -> list[str]:
    import questionary

    label_width = max((len(r.cell.label or r.cell.id) for r in rows), default=10)
    choices = []
    for row in rows:
        caption = (row.cell.label or row.cell.id).ljust(label_width)
        title = f"{caption}  {row.status_label}"
        choices.append(
            questionary.Choice(
                title=title,
                value=row.cell.id,
                checked=row.default_selected,
                disabled=None if row.selectable else row.gate.detail,
            )
        )
    selected = questionary.checkbox(
        "Select cells to run (space toggles, enter confirms):",
        choices=choices,
    ).ask()
    return list(selected) if selected else []
