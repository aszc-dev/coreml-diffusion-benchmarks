"""The run plan produced by `config` and consumed by `run`.

A small, durable record of the user's deliberate choices: which cells to run,
whether power measurement is on, log verbosity, repeat count for between-run
noise characterisation, and a free-form note about run conditions (R10.4).
Persisted as JSON under the workspace state dir.

Two presets cover ~95 % of intent:

* :func:`publication_preset` — the default ``sdbench`` is meant for. Repeats=7
  feeds the aggregator's median + p10/p90 over enough passes for
  ``validate-report`` to accept the energy spread, power on, full matrix.
* :func:`fast_test_preset` — quick iteration during development. Repeats=1,
  iterations clamped to the R5.3 floor (10), power off, verbosity quiet,
  collapses to a single cell so a turnaround under a minute is realistic.

The dataclass schema is backward-compatible: pre-multi-run plans load with
default ``repeats=1`` so they keep their original single-run behaviour.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

VERBOSITY_LEVELS = ("quiet", "normal", "verbose")
PLAN_MODES = ("publication", "fast", "custom")


@dataclass(frozen=True)
class RunPlan:
    cell_ids: list[str]
    power_enabled: bool
    verbosity: str = "normal"
    run_conditions: str = ""
    # Multi-run defaults. ``repeats > 1`` triggers ``run_session`` and the
    # per-cell aggregate (median + p10/p90); ``repeats == 1`` keeps the
    # single-run path. ``cooldown_s`` is the gate between passes (sleep +
    # thermal check); ``iterations`` overrides ``matrix.yaml`` per run when
    # set (used by the fast preset to clamp to the R5.3 floor without
    # touching the committed config). ``mode`` is the preset label, retained
    # so the TUI can show what was chosen without inferring it from numbers.
    repeats: int = 1
    cooldown_s: float = 30.0
    iterations: int | None = None
    mode: str = "custom"

    def __post_init__(self) -> None:
        if self.verbosity not in VERBOSITY_LEVELS:
            raise ValueError(f"verbosity must be one of {VERBOSITY_LEVELS}, got {self.verbosity!r}")
        if self.mode not in PLAN_MODES:
            raise ValueError(f"mode must be one of {PLAN_MODES}, got {self.mode!r}")
        if self.repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {self.repeats}")
        if self.iterations is not None and self.iterations < 10:
            raise ValueError(f"iterations override must be >= 10 (R5.3 floor), got {self.iterations}")


def publication_preset(
    cell_ids: list[str],
    *,
    power_enabled: bool,
    run_conditions: str = "",
) -> RunPlan:
    """The default — multi-run aggregate suitable for the publication bundle.

    Repeats=7 puts every cell over the ``n_runs_ok >= 3`` floor
    ``validate-report`` enforces, with four extra passes of margin so a single
    outlier weighs ~14 % instead of 20 % — tightening the p10/p90 energy band
    on cells like apple-ane that show a one-run spread. Iterations falls back
    to the matrix's own setting (the canonical 30) so within-pass sampling
    noise is already tamed. Cooldown of 30 s + the harness's own thermal gate
    keeps each pass starting from a comparable cold state."""
    return RunPlan(
        cell_ids=list(cell_ids),
        power_enabled=power_enabled,
        verbosity="normal",
        run_conditions=run_conditions,
        repeats=7,
        cooldown_s=30.0,
        iterations=None,
        mode="publication",
    )


def fast_test_preset(
    cell_ids: list[str],
    *,
    run_conditions: str = "",
) -> RunPlan:
    """Quick-iteration preset: one pass, the R5.3 minimum iterations, no power.

    Sized for a "did my change build" loop rather than a measurement: a single
    cell, 10 iterations, no power (skips the sudo prompt and the env check),
    verbosity quiet so the dashboard doesn't get in the way. Whatever the
    caller passes as ``cell_ids`` is trimmed to one entry — fast-test on the
    full matrix is a contradiction in terms; the first cell is the canonical
    pick because callers typically pass the result of ``full_suite_ids`` or a
    saved plan."""
    head = cell_ids[:1]
    return RunPlan(
        cell_ids=head,
        power_enabled=False,
        verbosity="quiet",
        run_conditions=run_conditions,
        repeats=1,
        cooldown_s=0.0,
        iterations=10,
        mode="fast",
    )


def save_runplan(plan: RunPlan, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_runplan(path: str | Path) -> RunPlan:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Defaults on read keep pre-multi-run plan files loading without a
    # migration step — the new fields land at their preset-neutral defaults.
    return RunPlan(
        cell_ids=list(data["cell_ids"]),
        power_enabled=bool(data["power_enabled"]),
        verbosity=data.get("verbosity", "normal"),
        run_conditions=data.get("run_conditions", ""),
        repeats=int(data.get("repeats", 1)),
        cooldown_s=float(data.get("cooldown_s", 30.0)),
        iterations=(int(data["iterations"]) if data.get("iterations") is not None else None),
        mode=data.get("mode", "custom"),
    )
