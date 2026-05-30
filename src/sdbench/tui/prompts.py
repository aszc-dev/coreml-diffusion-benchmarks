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
    """Every enabled and selectable cell — the deliberate FULL SUITE choice.

    ``enabled: false`` cells stay opt-in via the interactive picker. They were
    deliberately marked off by the matrix maintainer (capability outside the
    paper's scope, optional torch.compile path, MLX-native quantization point
    backends don't support yet — see ``config/matrix.yaml``); auto-running them
    from FULL SUITE puts ``failed`` rows in the publication bundle that read as
    incomplete work rather than the intentional N/A they are.
    """
    return [row.cell.id for row in rows if row.selectable and row.cell.enabled]


def default_ids(rows: list[CellRow]) -> list[str]:
    return [row.cell.id for row in rows if row.default_selected]


def select_cells_interactive(rows: list[CellRow], models: list[str] | None = None) -> list[str]:
    """Pick cells via a (model × cell) matrix.

    ``models`` is the list of model row labels; today's matrix has a single
    checkpoint (``SD 1.5``), but the grid is laid out so adding a row per
    extra checkpoint later is just an additional ``models`` entry — the
    selection logic, key bindings, and header layout do not change. The
    return value collapses to ``list[str]`` of cell ids while the matrix
    has only one model; a multi-model rewrite can promote it to a dict
    without disturbing today's callers.

    Falls back to questionary's checkbox in two cases: the terminal can't
    render multi-line headers (height < ~8) or prompt_toolkit cannot start
    (no controlling tty in CI). Both leave selection working, just less
    pretty."""
    models = models or ["SD 1.5"]
    import shutil

    cols, lines = shutil.get_terminal_size((80, 24))
    matrix_min_height = 8  # 4 header rows + 1 row per model + chrome
    needs_matrix = True
    if lines < matrix_min_height + len(models):
        needs_matrix = False
    try:
        if needs_matrix:
            return _matrix_picker(models, rows, terminal_cols=cols)
    except Exception:  # noqa: BLE001 — fall back rather than break selection.
        pass
    return _questionary_checkbox(rows)


def _questionary_checkbox(rows: list[CellRow]) -> list[str]:
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


# ---------------------------------------------------------------------------
# Matrix picker (model rows × cell columns)
# ---------------------------------------------------------------------------


_COL_WIDTH = 7  # "[x]" / "[ ]" / "—" centred, plus padding for header tokens
_ROW_LABEL_WIDTH = 22


def _column_header_tokens(row: CellRow) -> list[str]:
    """Four tokens per column, top to bottom: backend, compute_unit,
    attention, precision (with ``+cmp`` appended when the cell is the
    compile variant). Kept ≤ 6 chars each so a 9-column matrix fits in
    an 80-col terminal."""
    cell = row.cell
    backend = _abbrev_backend(cell.backend)
    cu = _abbrev_compute_unit(cell.compute_unit)
    attn = _abbrev_attention(cell.attention)
    precision = cell.precision
    if "compile" in (cell.id or "").lower() or "compile" in (cell.label or "").lower():
        precision = f"{precision}+c"
    return [backend, cu, attn, precision]


def _abbrev_backend(backend: str) -> str:
    return {
        "apple_coreml": "apple",
        "coreml_diffusion": "ours",
        "diffusers_mps": "diff",
        "mlx": "mlx",
    }.get(backend, backend[:6])


def _abbrev_compute_unit(cu: str) -> str:
    return {"CPU_AND_NE": "ANE", "CPU_AND_GPU": "GPU", "MPS": "MPS", "GPU": "GPU"}.get(cu, cu[:5])


def _abbrev_attention(attn: str) -> str:
    return {"SPLIT_EINSUM_V2": "SE2", "ORIGINAL": "ORIG", "NATIVE": "NATV"}.get(attn, attn[:5])


def _matrix_picker(models: list[str], rows: list[CellRow], terminal_cols: int) -> list[str]:
    """Render a (model × cell) grid that the user navigates with arrow keys.

    Single-model today, multi-model tomorrow. ``terminal_cols`` is consulted
    only to decide whether the headers fit; we never resize the column width
    because shrinking past 6 chars makes the precision token unreadable."""
    from prompt_toolkit import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    if not rows:
        return []
    selectable_cols = [j for j, r in enumerate(rows) if r.selectable]
    if not selectable_cols:
        return []

    # State: per (row_index, col_index) checked flag. Today's caller passes
    # one model row; the data model already supports many.
    checked: set[tuple[int, int]] = {
        (mi, ci)
        for mi in range(len(models))
        for ci, row in enumerate(rows)
        if row.default_selected
    }
    state: dict = {
        "row": 0,
        "col": selectable_cols[0],
        "result": None,
    }

    def _render() -> FormattedText:
        out: list[tuple[str, str]] = []
        # Four header bands.
        header_bands = [
            "backend",
            "engine",
            "attn",
            "prec",
        ]
        cols_by_band = list(zip(*[_column_header_tokens(r) for r in rows]))
        for band_label, tokens in zip(header_bands, cols_by_band):
            out.append(("class:band", band_label.rjust(_ROW_LABEL_WIDTH)))
            for token in tokens:
                out.append(("class:header", " " + token.center(_COL_WIDTH - 1)))
            out.append(("", "\n"))
        # Separator.
        out.append(("class:dim", "─" * (_ROW_LABEL_WIDTH + len(rows) * _COL_WIDTH + 1) + "\n"))
        # Model rows.
        for mi, model in enumerate(models):
            label = model[: _ROW_LABEL_WIDTH - 1].ljust(_ROW_LABEL_WIDTH)
            out.append(("class:model", label))
            for ci, row in enumerate(rows):
                is_cursor = (mi == state["row"] and ci == state["col"])
                if not row.selectable:
                    cell_str = "—"
                    style = "class:locked"
                else:
                    cell_str = "[x]" if (mi, ci) in checked else "[ ]"
                    style = "class:on" if (mi, ci) in checked else "class:off"
                if is_cursor:
                    style = "class:cursor"
                out.append((style, " " + cell_str.center(_COL_WIDTH - 1)))
            out.append(("", "\n"))
        # Footer hint.
        out.append(("class:dim", "\n"))
        out.append(
            (
                "class:hint",
                " ←→ ↑↓ move · space toggle · a row · A all · enter confirm · esc cancel\n",
            )
        )
        if state["row"] < len(models) and state["col"] < len(rows):
            current = rows[state["col"]]
            detail = current.cell.label or current.cell.id
            if not current.selectable:
                detail = f"{detail}  (locked: {current.gate.detail})"
            out.append(("class:hint", f" cursor: {models[state['row']]} × {detail}\n"))
        return FormattedText(out)

    kb = KeyBindings()

    def _move_col(delta: int) -> None:
        idx = state["col"]
        for _ in range(len(rows)):
            idx = (idx + delta) % len(rows)
            if rows[idx].selectable:
                state["col"] = idx
                return

    def _move_row(delta: int) -> None:
        if len(models) > 1:
            state["row"] = (state["row"] + delta) % len(models)

    @kb.add("left")
    @kb.add("h")
    def _(event): _move_col(-1)

    @kb.add("right")
    @kb.add("l")
    def _(event): _move_col(1)

    @kb.add("up")
    @kb.add("k")
    def _(event): _move_row(-1)

    @kb.add("down")
    @kb.add("j")
    def _(event): _move_row(1)

    @kb.add(" ")
    def _(event):
        key = (state["row"], state["col"])
        if not rows[state["col"]].selectable:
            return
        if key in checked:
            checked.remove(key)
        else:
            checked.add(key)

    @kb.add("a")
    def _(event):
        # Toggle every selectable column in the current model row.
        row_keys = [(state["row"], j) for j in selectable_cols]
        if all(k in checked for k in row_keys):
            for k in row_keys:
                checked.discard(k)
        else:
            for k in row_keys:
                checked.add(k)

    @kb.add("A")
    def _(event):
        # Select-all across every model row.
        all_keys = [(mi, j) for mi in range(len(models)) for j in selectable_cols]
        if all(k in checked for k in all_keys):
            checked.clear()
        else:
            for k in all_keys:
                checked.add(k)

    @kb.add("enter")
    def _(event):
        # Single-model: collapse to a list of cell ids for backward compat.
        if len(models) == 1:
            state["result"] = sorted({rows[ci].cell.id for (mi, ci) in checked if mi == 0})
        else:
            # Multi-model: return ids selected for ANY row (today no caller
            # consumes this, but the matrix is honest about it).
            state["result"] = sorted({rows[ci].cell.id for (_, ci) in checked})
        event.app.exit()

    @kb.add("c-c")
    @kb.add("escape")
    def _(event):
        state["result"] = []
        event.app.exit()

    body = FormattedTextControl(_render, focusable=True, show_cursor=False)
    layout = Layout(HSplit([Window(body)]))
    style = Style.from_dict(
        {
            "band": "italic #808080",
            "header": "bold",
            "model": "bold",
            "on": "ansigreen",
            "off": "",
            "locked": "#606060",
            "cursor": "reverse",
            "dim": "#606060",
            "hint": "#808080",
        }
    )
    app = Application(layout=layout, key_bindings=kb, style=style, full_screen=False, mouse_support=False)
    app.run()
    return state["result"] or []
