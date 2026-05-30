"""Review-fix regression tests: matrix overrides + harness git_sha fallback.

These three fixes share one motivation: the publication bundle must tell the
truth about what ran, including the cases where it diverged from the committed
source matrix or from the obvious provenance probe.
"""

import json
from pathlib import Path

import yaml

from sdbench.report import build_report, validate_report
from sdbench.telemetry import _resolve_harness_commit
from sdbench.tui.workspace import Workspace


def _write_workspace(
    tmp_path: Path,
    *,
    cells_run: list[str],
    source_cells: list[dict],
    matrix_overrides: list[str] | None = None,
    extra_manifest: dict | None = None,
) -> Workspace:
    ws = Workspace.resolve(tmp_path)
    ws.results_data_dir.mkdir(parents=True, exist_ok=True)
    ws.results_tables_dir.mkdir(parents=True, exist_ok=True)
    ws.results_raw_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 3,
        "run_id": "test-run",
        "cells_run": cells_run,
        "matrix_overrides": matrix_overrides or [],
        "provenance_digest": "deadbeef" * 8,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
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


def test_bundled_matrix_flags_override_cells(tmp_path):
    # Source says mlx-w8 is opt-in (enabled: false); the operator deliberately
    # ticked it for this run. The bundled matrix.yaml must mark it as an
    # override so cloning the bundle reproduces the same scope without the
    # reader having to cross-reference manifest.cells_run.
    source = [
        {"id": "alpha", "backend": "apple_coreml", "enabled": True},
        {"id": "mlx-w8", "backend": "mlx", "enabled": False},
    ]
    ws = _write_workspace(
        tmp_path,
        cells_run=["alpha", "mlx-w8"],
        source_cells=source,
        matrix_overrides=["mlx-w8"],
    )

    bundle = build_report(ws, output_root=tmp_path / "out", zip_bundle=False)
    realised_text = (bundle / "matrix.yaml").read_text(encoding="utf-8")
    realised = yaml.safe_load(realised_text)

    # Override cells are forced enabled but tagged with _matrix_override so the
    # difference from matrix.source.yaml is impossible to miss.
    by_id = {c["id"]: c for c in realised["cells"]}
    assert by_id["mlx-w8"]["enabled"] is True
    assert by_id["mlx-w8"].get("_matrix_override") is True
    assert by_id["alpha"].get("_matrix_override") is not True
    # Plus a header comment that lists overrides by id so a yaml-blind reader
    # spots them in `cat matrix.yaml` without grokking the yaml structure.
    header_block = "\n".join(
        line for line in realised_text.splitlines() if line.startswith("#")
    )
    assert "matrix_overrides" in header_block
    assert "mlx-w8" in header_block


def test_validate_report_surfaces_overrides_without_failing(tmp_path):
    # Overrides are informational — the bundle records them properly, so the
    # validator must NOT mark the bundle as failed just because overrides
    # exist (otherwise every legitimate experiment run breaks validation).
    source = [
        {"id": "alpha", "backend": "apple_coreml", "enabled": True},
        {"id": "mlx-w8", "backend": "mlx", "enabled": False},
    ]
    ws = _write_workspace(
        tmp_path,
        cells_run=["alpha", "mlx-w8"],
        source_cells=source,
        matrix_overrides=["mlx-w8"],
        extra_manifest={
            "determinism": {
                "latent_sha256": "a" * 64,
                "text_embedding_sha256": "b" * 64,
            },
        },
    )
    # results.jsonl with full BenchmarkRecord shape (load_jsonl re-instantiates
    # the dataclass) and matching digests so the unrelated digest checks pass.
    from sdbench.results import BenchmarkRecord, write_jsonl

    records = [
        BenchmarkRecord(
            run_id="test-run",
            cell_id=cid,
            backend="apple_coreml",
            requested_compute_unit="CPU_AND_NE",
            realized_compute_unit="CPU_AND_NE",
            attention="SPLIT_EINSUM_V2",
            precision="fp16",
            resolution=512,
            status="ok",
            latency_ms_median=400.0,
            latency_ms_iqr=0.5,
            gpu_power_w=None,
            ane_power_w=None,
            energy_per_unet_step_j=None,
            estimated_energy_per_50_step_image_j=None,
            mse=1.0e-5,
            cosine=0.9999,
            numerically_divergent=False,
            on_disk_size_bytes=1024,
            weight_only_size_bytes=1024,
            effective_bits_per_parameter=16.0,
            compute_precision="fp16",
            graph_capture_s=None,
            convert_s=None,
            first_load_compile_s=None,
            failure_reason=None,
            provenance_digest="deadbeef" * 8,
            latent_input_sha256="a" * 64,
            text_embedding_input_sha256="b" * 64,
        )
        for cid in ("alpha", "mlx-w8")
    ]
    write_jsonl(records, ws.results_data_dir / "results.jsonl")

    bundle = build_report(ws, output_root=tmp_path / "out", zip_bundle=False)
    result = validate_report(bundle)
    assert result.matrix_overrides == ["mlx-w8"]
    assert result.ok is True  # informational, not a fail


def test_resolve_harness_commit_falls_back_to_package_path(monkeypatch):
    # No build stamp, no PEP 610, no workspace .git — we must still report a
    # SHA when the sdbench package itself sits in a git tree on disk, which is
    # the "ran from a subdirectory of the clone" case the user hit. ``test_telemetry.py``
    # covers the build_stamp/pep610/workspace preferences; this is the new
    # fourth-tier fallback.
    from sdbench import _build_info, telemetry

    monkeypatch.setattr(_build_info, "BUILD_GIT_SHA", None, raising=False)
    monkeypatch.setattr(_build_info, "BUILD_GIT_DESCRIBE", None, raising=False)
    monkeypatch.setattr(telemetry, "_read_pep610_commit", lambda: None)
    monkeypatch.setattr(telemetry, "_probe_package_git", lambda: ("pkg-sha", "pkg-describe"))

    sha, describe, source = _resolve_harness_commit(None, None)
    assert sha == "pkg-sha"
    assert describe == "pkg-describe"
    assert source == "package_path"


def test_resolve_harness_commit_returns_none_source_when_nothing_available(monkeypatch):
    from sdbench import _build_info, telemetry

    monkeypatch.setattr(_build_info, "BUILD_GIT_SHA", None, raising=False)
    monkeypatch.setattr(_build_info, "BUILD_GIT_DESCRIBE", None, raising=False)
    monkeypatch.setattr(telemetry, "_read_pep610_commit", lambda: None)
    monkeypatch.setattr(telemetry, "_probe_package_git", lambda: (None, None))

    sha, describe, source = _resolve_harness_commit(None, None)
    assert sha is None
    assert describe is None
    assert source == "none"


def test_probe_package_git_returns_sha_from_real_clone():
    # We *are* running inside the harness clone right now — the probe should
    # find a SHA without any patching. This is the integration check that the
    # fallback actually does what it claims when invoked from a checkout.
    from sdbench.telemetry import _probe_package_git

    sha, describe = _probe_package_git()
    assert sha is not None and len(sha) >= 7
    # describe is allowed to be None (no tags reachable), but if present it
    # should be a non-empty string.
    assert describe is None or isinstance(describe, str) and describe
