"""Progress reporting hooks for a matrix run.

The orchestrator and timing loop call these hooks so a front-end can render live
progress without the harness importing Rich or knowing anything about display.
The default implementation is a no-op, so headless/scripted runs are unaffected.
"""


class RunReporter:
    """No-op base. Override the hooks you care about."""

    # When True, run_benchmark redirects backend stdout/stderr into reporter.log
    # (so in-process library output lands in the UI instead of corrupting a
    # full-screen Live). Line reporters leave it False and let output print.
    capture_output = False

    def run_start(self, total_cells: int) -> None: ...

    def cell_start(self, cell_id: str, index: int, total: int) -> None: ...

    def cell_prepared(self, cell_id: str, compute_unit: str) -> None: ...

    def warmup_step(self, cell_id: str, index: int, total: int, latency_ms: float) -> None: ...

    def timed_step(self, cell_id: str, index: int, total: int, latency_ms: float) -> None: ...

    def cell_done(self, record) -> None: ...

    def cell_failed(self, cell_id: str, reason: str) -> None: ...

    def log(self, message: str) -> None:
        """Free-form status line (power auth, invalidation, alignment, …)."""
        ...

    def run_done(self, records: list) -> None: ...


class NullReporter(RunReporter):
    pass
