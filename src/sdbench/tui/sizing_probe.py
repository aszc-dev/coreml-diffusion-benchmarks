"""Measure the on-disk footprint of converted artifacts — measured, never guessed.

`config/disk_footprint.yaml` is populated from real measurements after a
conversion, so the config UI can warn about required disk space honestly instead
of estimating (the spec demands the full shipped artifact size, R8.2).
"""

import time
from pathlib import Path

import yaml

from sdbench.backends.apple_coreml import AppleCoreMLAdapter
from sdbench.backends.coreml_diffusion import CoreMLDiffusionAdapter
from sdbench.config import CellConfig
from sdbench.sizing import artifact_size_bytes


def _artifact_for(cell: CellConfig, apple: AppleCoreMLAdapter, team: CoreMLDiffusionAdapter) -> Path | None:
    try:
        if cell.backend == "apple_coreml":
            return apple._artifact_path(cell)
        if cell.backend == "coreml_diffusion":
            return team._artifact_path(cell)
    except ValueError:
        return None
    return None


def measure_cell_footprint(ws, cfg) -> dict[str, int]:
    """Measured artifact size per cell, for cells whose converted artifact exists."""
    apple = AppleCoreMLAdapter("unused", artifact_root=ws.artifacts_dir / "apple_coreml")
    team = CoreMLDiffusionAdapter("unused", artifact_root=ws.artifacts_dir / "coreml_diffusion")
    sizes: dict[str, int] = {}
    for cell in cfg.cells:
        artifact = _artifact_for(cell, apple, team)
        if artifact is not None and artifact.exists():
            sizes[cell.id] = artifact_size_bytes(artifact)
    return sizes


def write_footprint(path: str | Path, sizes: dict[str, int]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cells": dict(sorted(sizes.items())),
    }
    output.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
