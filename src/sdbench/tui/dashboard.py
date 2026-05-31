"""Full-screen dashboards for run and convert.

Both show the *whole* list of items (cells / builds) with a per-item status, not
just the one in flight, plus an overall progress bar and a streaming log panel.
Everything renders inside the app's single Live (the Progress is a renderable, not
its own Live — a nested Live causes newline spam). Panels are stacked so each gets
the full terminal width.
"""

from collections import deque

from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from sdbench.progress import RunReporter
from sdbench.tui import screen
from sdbench.tui.run_view import equivalence_plain, fmt_ms, summary_text

_MARK = {
    "pending": ("·", "sdbench.dim"),
    "running": ("▶", "sdbench.title"),
    "building": ("▶", "sdbench.title"),
    "done": ("✓", "sdbench.ok"),
    "cached": ("=", "sdbench.dim"),
    "failed": ("✗", "sdbench.danger"),
}


def _overall_bar() -> Progress:
    # bar_width=None lets the bar absorb whatever horizontal space the panel
    # has after the fixed-width text columns — otherwise Rich's 40-cell default
    # leaves dead space to the right on wide terminals.
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        expand=True,
    )


def _items_table(rows: list[tuple[str, str, str]]) -> Table:
    """rows = (name, status, detail). One line per item, full list always visible."""
    table = Table(expand=True, box=None, show_header=False, pad_edge=False)
    table.add_column(width=2)
    table.add_column(ratio=1, no_wrap=True)
    table.add_column(justify="right", no_wrap=True)
    for name, status, detail in rows:
        mark, style = _MARK.get(status, ("·", "sdbench.dim"))
        table.add_row(Text(mark, style=style), Text(name), Text(detail or status, style=style))
    return table


def _logs_panel(logs: deque, title: str, tail: int = 12) -> Panel:
    # no_wrap + ellipsis overflow guarantees no log line ever wraps to a second
    # row and no excess height escapes the region — otherwise long backend lines
    # push the panel taller, the layout overflows the alt screen, and the
    # terminal scrolls (the "lines sliding from under the top edge" symptom).
    text = Text(no_wrap=True, overflow="ellipsis")
    for line in list(logs)[-tail:]:
        # Honour ``[sdbench.warn]…[/]`` markup so the few lines that deserve
        # attention (power-env refusal, thermal abort, sudo declined) actually
        # stand out instead of blending into the backend chatter.
        text.append_text(Text.from_markup(line))
        text.append("\n")
    return Panel(text, title=title, border_style="sdbench.dim", height=tail + 2)


class DashboardReporter(RunReporter):
    capture_output = True  # backend stdout/stderr -> the Logs panel, not the terminal

    def __init__(self, live, ws, cell_ids=None, title: str = "sdbench · run") -> None:
        self.live = live
        self.ws = ws
        self.title = title
        self.progress = _overall_bar()
        self._task = None
        self.order: list[str] = list(cell_ids or [])
        self.status: dict[str, str] = {cid: "pending" for cid in self.order}
        self.detail: dict[str, str] = {cid: "" for cid in self.order}
        self._logs: deque[str] = deque(maxlen=500)
        self._summary: Text | None = None
        # Multi-run session bookkeeping. ``run_matrix`` fires ``run_done`` after
        # every pass; without this the pass-1 summary latches and freezes the
        # screen while passes 2..N run underneath (any-key reads nothing until
        # the whole session returns). begin_session is duck-typed — only the
        # session driver calls it; single-pass callers leave repeats=1.
        self._repeats = 1
        self._pass = 0

    def begin_session(self, repeats: int) -> None:
        self._repeats = max(1, repeats)
        self._pass = 0

    def run_start(self, total_cells: int) -> None:
        # New pass: drop any latched summary so live progress shows again, and
        # reset per-cell state + the progress bar instead of stacking a fresh
        # task on top of the previous pass's bar.
        self._pass += 1
        self._summary = None
        for cid in self.order:
            self.status[cid] = "pending"
            self.detail[cid] = ""
        label = "Matrix" if self._repeats == 1 else f"Matrix · pass {self._pass}/{self._repeats}"
        if self._task is None:
            self._task = self.progress.add_task(label, total=total_cells)
        else:
            self.progress.reset(self._task, total=total_cells, description=label)
        self._refresh()

    def cell_start(self, cell_id: str, index: int, total: int) -> None:
        if cell_id not in self.status:
            self.order.append(cell_id)
        self.status[cell_id] = "running"
        self.detail[cell_id] = "starting…"
        self.log(f"[{index + 1}/{total}] {cell_id} — running")

    def cell_prepared(self, cell_id: str, compute_unit: str) -> None:
        self.detail[cell_id] = f"prepared · {compute_unit}"
        self._refresh()

    def warmup_step(self, cell_id: str, index: int, total: int, latency_ms: float) -> None:
        self.detail[cell_id] = f"warmup {index}/{total}"
        self._refresh()

    def timed_step(self, cell_id: str, index: int, total: int, latency_ms: float) -> None:
        self.detail[cell_id] = f"timed {index}/{total}"
        self._refresh()

    def cell_done(self, record) -> None:
        self.status[record.cell_id] = "done" if record.status == "ok" else "failed"
        self.detail[record.cell_id] = f"{fmt_ms(record.latency_ms_median)} ms · {equivalence_plain(record)}"
        self._advance()
        self.log(f"  ✓ {record.cell_id} · {self.detail[record.cell_id]}")

    def cell_failed(self, cell_id: str, reason: str) -> None:
        self.status[cell_id] = "failed"
        self.detail[cell_id] = reason
        self._advance()
        self.log(f"  ✗ {cell_id} · {reason}")

    def log(self, message: str) -> None:
        self._logs.append(message)
        self._refresh()

    def run_done(self, records: list) -> None:
        # Only latch the summary on the final pass; intermediate passes must
        # leave the live view up so the next pass (and the cooldown between
        # them) keeps rendering.
        if self._pass >= self._repeats:
            self.show_summary(records)

    def show_summary(self, records: list) -> None:
        self._summary = summary_text(records)
        self._refresh()

    def _advance(self) -> None:
        if self._task is not None:
            self.progress.advance(self._task)

    def render(self) -> Layout:
        if self._summary is not None:
            return screen.frame(
                screen.header(self.title, Text("benchmark running", style="sdbench.dim"), screen.usage_text(self.ws)),
                Panel(self._summary, title="Run summary", border_style="sdbench.title"),
                screen.footer("done · results upserted · press any key to return"),
            )
        rows = [(cid, self.status[cid], self.detail.get(cid, "")) for cid in self.order]
        body = Layout()
        body.split_column(
            Layout(Panel(self.progress, title="Progress", border_style="sdbench.dim"), name="bar", size=3),
            Layout(Panel(_items_table(rows), title="Cells", border_style="sdbench.dim"), name="cells", size=min(len(rows) + 3, 16)),
            Layout(_logs_panel(self._logs, "Logs", tail=12), name="logs", ratio=1),
        )
        return screen.frame(
            screen.header(self.title, Text("benchmark running", style="sdbench.dim"), screen.usage_text(self.ws)),
            body,
            screen.footer("running… (only the powermetrics sampler runs as root)"),
        )

    def _refresh(self) -> None:
        # Only update state; Rich's auto-refresh thread (started by live_screen)
        # is the sole drawer. Multiple threads calling live.refresh() concurrently
        # desync the cursor — keep this as update-only.
        self.live.update(self.render())


class ConvertDashboard:
    """Conversion view: overall bar + full builds list (with status) + streaming logs."""

    def __init__(self, live, ws, builds, title: str = "sdbench · convert") -> None:
        self.live = live
        self.ws = ws
        self.title = title
        self.builds = list(builds)
        self.progress = _overall_bar()
        self._task = self.progress.add_task("Converting", total=max(len(self.builds), 1))
        self.status: dict[str, str] = {self._key(b): "pending" for b in self.builds}
        self._logs: deque[str] = deque(maxlen=2000)
        self._current: str | None = None
        self._done = False
        self._refresh()

    @staticmethod
    def _key(build) -> str:
        return str(build.expected_artifact)

    @staticmethod
    def _label(build) -> str:
        return f"{build.backend} · {build.output_dir.name}"

    def on_skip(self, build) -> None:
        self.status[self._key(build)] = "cached"
        self._advance()
        self._logs.append(f"cached · {self._label(build)}")
        self._refresh()

    def on_build(self, build, index: int, total: int) -> None:
        self._current = self._key(build)
        self.status[self._current] = "building"
        self._logs.append(f"── building {self._label(build)}  ({index + 1}/{total}) ──")
        self._refresh()

    def on_done(self, build) -> None:
        self.status[self._key(build)] = "done"
        self._current = None
        self._advance()
        self._refresh()

    def on_line(self, text: str) -> None:
        self._logs.append(text)
        self._refresh()

    def on_error(self, message: str) -> None:
        if self._current is not None:
            self.status[self._current] = "failed"
        self._logs.append(f"ERROR: {message}")
        self._refresh()

    def finish(self) -> None:
        self._done = True
        self._refresh()

    def _advance(self) -> None:
        done = sum(1 for status in self.status.values() if status in ("done", "cached", "failed"))
        self.progress.update(self._task, completed=done)

    def render(self) -> Layout:
        rows = [(self._label(b), self.status[self._key(b)], "") for b in self.builds]
        body = Layout()
        body.split_column(
            Layout(Panel(self.progress, title="Converting", border_style="sdbench.dim"), name="bar", size=3),
            Layout(Panel(_items_table(rows), title="Builds", border_style="sdbench.dim"), name="builds", size=min(len(rows) + 3, 14)),
            Layout(_logs_panel(self._logs, "Toolchain logs", tail=14), name="logs", ratio=1),
        )
        hint = "done · press any key to return" if self._done else "converting in the isolated ct8 / ct9 envs…"
        return screen.frame(
            screen.header(self.title, Text("conversion running", style="sdbench.dim"), screen.usage_text(self.ws)),
            body,
            screen.footer(hint),
        )

    def _refresh(self) -> None:
        # Update only; Rich's auto-refresh thread is the sole drawer.
        self.live.update(self.render())
