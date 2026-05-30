"""Smoke tests for the report bundling matrix.yaml provenance.

The bundle's ``matrix.yaml`` used to be a verbatim copy of the source file,
which left ``enabled: false`` rows visible alongside a manifest ``cells_run``
list that disagreed with them. The new behaviour rewrites ``matrix.yaml`` to
match ``cells_run`` exactly and keeps the source under ``matrix.source.yaml``.
"""

import json
from pathlib import Path

import yaml

from sdbench.report import build_report
from sdbench.tui.workspace import Workspace


def _write_workspace(tmp_path: Path, cells_run: list[str], source_cells: list[dict]) -> Workspace:
    ws = Workspace.resolve(tmp_path)
    ws.results_data_dir.mkdir(parents=True, exist_ok=True)
    ws.results_tables_dir.mkdir(parents=True, exist_ok=True)
    ws.results_raw_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "run_id": "test-run",
        "cells_run": cells_run,
        "provenance_digest": "deadbeef" * 8,
    }
    (ws.results_data_dir / "environment.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    matrix = {
        "checkpoint": "/x.safetensors",
        "seed": 0,
        "iterations": 10,
        "cells": source_cells,
    }
    ws.matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    return ws


def test_bundle_matrix_reflects_cells_run_not_source_file(tmp_path):
    source_cells = [
        {"id": "alpha", "backend": "apple_coreml", "enabled": True},
        {"id": "beta", "backend": "mlx", "enabled": False},      # disabled in source
        {"id": "gamma", "backend": "diffusers_mps", "enabled": True},
    ]
    # Operator selected just alpha + gamma at the CLI, so the run skipped beta.
    ws = _write_workspace(tmp_path, cells_run=["alpha", "gamma"], source_cells=source_cells)

    bundle = build_report(ws, output_root=tmp_path / "out", zip_bundle=False)
    realised = yaml.safe_load((bundle / "matrix.yaml").read_text(encoding="utf-8"))
    source_copy = yaml.safe_load((bundle / "matrix.source.yaml").read_text(encoding="utf-8"))

    assert [c["id"] for c in realised["cells"]] == ["alpha", "gamma"]
    assert all(c["enabled"] for c in realised["cells"])
    # The original is kept verbatim so a reviewer can still audit the input.
    assert [c["id"] for c in source_copy["cells"]] == ["alpha", "beta", "gamma"]
    assert source_copy["cells"][1]["enabled"] is False


def test_bundle_synthesises_cells_absent_from_source(tmp_path):
    """A CLI ``run-cell --backend ... --compute-unit ...`` invocation can run a
    cell that isn't in the static matrix; the realised yaml should still list it."""
    source_cells = [{"id": "alpha", "backend": "apple_coreml", "enabled": True}]
    ws = _write_workspace(tmp_path, cells_run=["alpha", "adhoc"], source_cells=source_cells)

    bundle = build_report(ws, output_root=tmp_path / "out", zip_bundle=False)
    realised = yaml.safe_load((bundle / "matrix.yaml").read_text(encoding="utf-8"))

    ids = [c["id"] for c in realised["cells"]]
    assert ids == ["alpha", "adhoc"]
    adhoc = next(c for c in realised["cells"] if c["id"] == "adhoc")
    assert adhoc.get("_synthesised") is True
