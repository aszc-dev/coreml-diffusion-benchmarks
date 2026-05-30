"""Line-based run reporter for the CLI (implements RunReporter).

Plain, ordered console output that renders correctly in any terminal: no live
cursor control, so it can never spam newlines or scramble. The full-screen
dashboard used by the guided app lives in `tui/dashboard.py`; both share the
equivalence/format helpers here.
"""

from rich.text import Text

from sdbench.progress import RunReporter
from sdbench.tui.console import console


class SimpleReporter(RunReporter):
    def __init__(self, verbosity: str = "normal") -> None:
        self.verbosity = verbosity
        self._total = 0
        self._index = 0

    def run_start(self, total_cells: int) -> None:
        self._total = total_cells
        if self.verbosity != "quiet":
            console.rule(f"[sdbench.title]Running {total_cells} cell(s)[/]")

    def cell_start(self, cell_id: str, index: int, total: int) -> None:
        self._index = index + 1
        if self.verbosity != "quiet":
            console.print(f"[sdbench.dim][{self._index}/{self._total}] {cell_id} — running…[/]")

    def warmup_step(self, cell_id: str, index: int, total: int, latency_ms: float) -> None:
        if self.verbosity == "verbose":
            console.print(f"[sdbench.dim]      warmup {index}/{total} · {latency_ms:.0f} ms[/]")

    def timed_step(self, cell_id: str, index: int, total: int, latency_ms: float) -> None:
        if self.verbosity == "verbose":
            console.print(f"[sdbench.dim]      timed  {index}/{total} · {latency_ms:.0f} ms[/]")

    def cell_done(self, record) -> None:
        if self.verbosity == "quiet":
            return
        latency = f"{record.latency_ms_median:.1f} ms" if record.latency_ms_median is not None else "n/a"
        console.print(f"  [sdbench.ok]✓[/] {record.cell_id} · {latency}{equivalence_markup(record)}")

    def cell_failed(self, cell_id: str, reason: str) -> None:
        if self.verbosity != "quiet":
            console.print(f"  [sdbench.danger]✗[/] {cell_id} · {reason}")

    def log(self, message: str) -> None:
        if self.verbosity != "quiet":
            console.print(f"[sdbench.dim]{message}[/]")

    def run_done(self, records: list) -> None:
        render_summary(records)


def _row_style(record) -> str:
    if record.status != "ok":
        return "sdbench.danger"
    if record.numerically_divergent:
        return "sdbench.warn"
    return "sdbench.ok"


def summary_text(records: list) -> Text:
    """Aligned run summary as a renderable (shared by the line reporter and dashboard)."""
    width = max([len(r.cell_id) for r in records] + [len("Cell")]) + 2  # fit the longest cell id
    text = Text()
    text.append(f"  {'Cell':<{width}}{'Status':<8}{'Median':>11}{'GPU W':>8}{'ANE W':>8}  Equivalence\n", style="sdbench.dim")
    for record in records:
        row = (
            f"  {record.cell_id:<{width}}{record.status:<8}{fmt_ms(record.latency_ms_median):>11}"
            f"{fmt_w(record.gpu_power_w):>8}{fmt_w(record.ane_power_w):>8}  {equivalence_plain(record)}\n"
        )
        text.append(row, style=_row_style(record))
    if any(r.numerically_divergent for r in records):
        text.append(
            "  'flagged' = differs from the fp32 PyTorch reference beyond threshold; "
            "cosine shows how small the deviation is (precision difference, not a failure).",
            style="sdbench.dim",
        )
    return text


def render_summary(records: list) -> None:
    console.rule("[sdbench.title]Run summary[/]")
    console.print(summary_text(records), highlight=False)


def equivalence_markup(record) -> str:
    """Inline equivalence with markup: cosine value, flagged when over threshold."""
    if record.cosine is None:
        return ""
    cos = f"cos {record.cosine:.7f}"
    if record.numerically_divergent:
        mse = f", mse {record.mse:.2e}" if record.mse is not None else ""
        return f" · [sdbench.warn]{cos}{mse} (flagged)[/]"
    return f" · [sdbench.ok]{cos}[/]"


def equivalence_plain(record) -> str:
    """Unstyled equivalence for aligned rows."""
    if record.cosine is None:
        return "—"
    cos = f"cos {record.cosine:.7f}"
    if record.numerically_divergent:
        mse = f" mse {record.mse:.2e}" if record.mse is not None else ""
        return f"{cos}{mse} (flagged)"
    return cos


def fmt_ms(value) -> str:
    return f"{value:.1f}" if value is not None else "—"


def fmt_w(value) -> str:
    return f"{value:.2f}" if value is not None else "—"


# Back-compat alias for tests/imports that referenced the private helper name.
_equivalence = equivalence_markup
