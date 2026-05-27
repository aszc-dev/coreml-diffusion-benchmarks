import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


@dataclass(frozen=True)
class CellConfig:
    id: str
    backend: str
    compute_unit: str
    attention: str
    precision: str
    resolution: int
    label: str | None = None
    enabled: bool = True
    requires: dict[str, Any] | None = None
    notes: str | None = None


@dataclass(frozen=True)
class EquivalenceConfig:
    mse_max: float
    cosine_min: float
    reference: dict[str, Any] | None = None


@dataclass(frozen=True)
class PowerConfig:
    interval_ms: int
    baseline_seconds: float
    samplers: list[str] | None = None


@dataclass(frozen=True)
class ThermalConfig:
    throttle_policy: str = "abort"
    abort_on_throttle: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    checkpoint: Path
    seed: int
    iterations: int
    warmup: int
    thermal: ThermalConfig
    equivalence: EquivalenceConfig
    power: PowerConfig
    cells: list[CellConfig]

    def select_cell(
        self,
        backend: str,
        compute_unit: str,
        attention: str,
        precision: str,
        resolution: int,
    ) -> CellConfig:
        matches = [
            cell
            for cell in self.cells
            if cell.backend == backend
            and cell.compute_unit == compute_unit
            and cell.attention == attention
            and cell.precision == precision
            and cell.resolution == resolution
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one matching matrix cell, found {len(matches)}")
        return matches[0]

    def select_cell_by_id(self, cell_id: str) -> CellConfig:
        matches = [cell for cell in self.cells if cell.id == cell_id]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one matrix cell with id={cell_id}, found {len(matches)}")
        return matches[0]

    def enabled_cells(self) -> list[CellConfig]:
        return [cell for cell in self.cells if cell.enabled]


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    env = _load_dotenv(Path.cwd() / ".env") | _load_dotenv(config_path.parent / ".env") | os.environ
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expanded = _expand_env(raw, env)
    iterations = int(expanded["iterations"])
    if iterations < 10:
        raise ValueError("Benchmark must run at least 10 timed iterations")

    thermal_raw = expanded.get("thermal", {})
    throttle_policy = thermal_raw.get("throttle_policy")
    abort_on_throttle = bool(thermal_raw.get("abort_on_throttle", throttle_policy != "flag"))
    return BenchmarkConfig(
        checkpoint=Path(expanded["checkpoint"]).expanduser(),
        seed=int(expanded["seed"]),
        iterations=iterations,
        warmup=int(expanded.get("warmup", 1)),
        thermal=ThermalConfig(
            throttle_policy=throttle_policy or ("abort" if abort_on_throttle else "flag"),
            abort_on_throttle=abort_on_throttle,
        ),
        equivalence=EquivalenceConfig(**expanded["equivalence"]),
        power=PowerConfig(**expanded["power"]),
        cells=[CellConfig(**_normalize_cell(cell, expanded)) for cell in expanded["cells"]],
    )


def _normalize_cell(cell: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(cell)
    normalized.setdefault("resolution", int(config.get("resolution_default", 512)))
    normalized.setdefault("enabled", True)
    return normalized


def _expand_env(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item, env) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item, env) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name = match.group(1)
        if name not in env:
            raise ValueError(f"Environment variable {name} is not set")
        return env[name]

    return ENV_PATTERN.sub(replace, value)


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values
