import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# POSIX-style `${VAR}` and `${VAR:-default}`. The fallback lets the canonical
# matrix.yaml ship with `${SD15_CHECKPOINT:-<cache path>}` so a fresh
# `uv tool install`-ed `cdbench` doesn't need any env var pre-set to load the
# config — the download flow then materialises the cache path.
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


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
    if not config_path.exists():
        # Materialise the packaged canonical matrix.yaml next to the requested
        # path so a fresh `uv tool install`-ed `cdbench` (no repo checkout) just
        # works. The repo's own `config/matrix.yaml` is the source the wheel
        # bundles; here we copy it back out into the user's workspace.
        materialise_packaged_matrix(config_path)
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
        fallback = match.group(2)
        if name in env:
            return env[name]
        if fallback is not None:
            return fallback
        raise ValueError(f"Environment variable {name} is not set")

    return ENV_PATTERN.sub(replace, value)


def materialise_packaged_matrix(target: Path) -> Path:
    """Copy the wheel-bundled ``matrix.yaml`` to ``target`` and return the path.

    Used by :func:`load_benchmark_config` so a fresh
    ``uv tool install coreml-diffusion-benchmarks`` followed by ``cdbench`` in
    an empty directory finds a working config without the contributor having
    to clone the repo. The wheel ships the repo's own ``config/matrix.yaml``
    at ``sdbench/data/matrix.yaml`` (see ``[tool.hatch.build.targets.wheel.force-include]``).
    """
    from importlib.resources import files

    target.parent.mkdir(parents=True, exist_ok=True)
    source = files("sdbench").joinpath("data/matrix.yaml")
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


# Packaged-resource -> workspace-relative-path map for the conversion toolchain.
# The wheel ships these under sdbench/data/; the bootstrap materialises them
# into the workspace tree so `uv run --project envs/apple-ct8` and
# `scripts/convert/apple_ct8.py` resolve relative to cwd without a repo
# checkout. The matrix.yaml entry is handled separately by
# materialise_packaged_matrix.
PACKAGED_CONVERT_TREE: tuple[tuple[str, str], ...] = (
    ("data/scripts/convert/apple_ct8.py", "scripts/convert/apple_ct8.py"),
    ("data/scripts/convert/team_ct9.py", "scripts/convert/team_ct9.py"),
    ("data/envs/apple-ct8/pyproject.toml", "envs/apple-ct8/pyproject.toml"),
    ("data/envs/apple-ct8/uv.lock", "envs/apple-ct8/uv.lock"),
    ("data/envs/team-ct9/pyproject.toml", "envs/team-ct9/pyproject.toml"),
    ("data/envs/team-ct9/uv.lock", "envs/team-ct9/uv.lock"),
)


def materialise_convert_tree(workspace_root: Path) -> list[Path]:
    """Materialise the conversion drivers + isolated env definitions under ``workspace_root``.

    Idempotent: skips files that already exist (so a repo checkout, which has
    them committed, is a no-op). Returns the list of paths that were written.

    The wheel ships the canonical files under ``sdbench/data/scripts/...`` and
    ``sdbench/data/envs/...``; dev mode picks them up through symlinks under
    ``src/sdbench/data/`` so there is exactly one source of truth.
    """
    from importlib.resources import files

    pkg = files("sdbench")
    written: list[Path] = []
    for resource_path, workspace_rel in PACKAGED_CONVERT_TREE:
        target = workspace_root / workspace_rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Use binary read so uv.lock (text) and any future binary go through one
        # path; symlinks in dev mode resolve transparently through importlib.
        target.write_bytes(pkg.joinpath(resource_path).read_bytes())
        written.append(target)
    return written


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
