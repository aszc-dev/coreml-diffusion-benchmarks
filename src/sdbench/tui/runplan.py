"""The run plan produced by `config` and consumed by `run`.

A small, durable record of the user's deliberate choices: which cells to run,
whether power measurement is on, log verbosity, and a free-form note about run
conditions (R10.4). Persisted as JSON under the workspace state dir.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

VERBOSITY_LEVELS = ("quiet", "normal", "verbose")


@dataclass(frozen=True)
class RunPlan:
    cell_ids: list[str]
    power_enabled: bool
    verbosity: str = "normal"
    run_conditions: str = ""

    def __post_init__(self) -> None:
        if self.verbosity not in VERBOSITY_LEVELS:
            raise ValueError(f"verbosity must be one of {VERBOSITY_LEVELS}, got {self.verbosity!r}")


def save_runplan(plan: RunPlan, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_runplan(path: str | Path) -> RunPlan:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunPlan(
        cell_ids=list(data["cell_ids"]),
        power_enabled=bool(data["power_enabled"]),
        verbosity=data.get("verbosity", "normal"),
        run_conditions=data.get("run_conditions", ""),
    )
