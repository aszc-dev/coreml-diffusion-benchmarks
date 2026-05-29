"""Tests for the assembled EnvironmentManifest and the contributor bundle.

Heavy probes (sysctl, git, uv, system_profiler) are NOT faked here — the manifest
is exercised against the real environment because at this layer the contract is
"populate every field that can be probed". Probe failures are accepted, but the
structural shape (schema_version, hardware block, determinism block, legacy
mirror fields) MUST hold.
"""

import json
import zipfile
from dataclasses import asdict

from sdbench import inputs
from sdbench.env import collect_environment_manifest, write_environment_manifest
from sdbench.report import build_report, validate_report
from sdbench.results import BenchmarkRecord, write_jsonl, write_summary_tables
from sdbench.telemetry import TELEMETRY_SCHEMA_VERSION
from sdbench.tui.workspace import Workspace


def _seed_workspace(tmp_path) -> Workspace:
    ws = Workspace.resolve(tmp_path)
    ws.results_data_dir.mkdir(parents=True, exist_ok=True)
    ws.results_tables_dir.mkdir(parents=True, exist_ok=True)
    ws.results_raw_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "matrix.yaml").write_text("checkpoint: dummy\n", encoding="utf-8")
    return ws


def test_manifest_carries_schema_version_and_blocks(tmp_path):
    ws = _seed_workspace(tmp_path)
    shared = inputs.generate_shared_input(seed=42, resolution=64)
    manifest = collect_environment_manifest(
        seed=42,
        run_conditions="test",
        checkpoint_path=None,
        workspace=ws,
        provenance_digest="dead",
        run_id="abc-run",
        shared_input=shared,
        shared_input_path=None,
        cells_run=["cell-a"],
    )
    assert manifest.schema_version == TELEMETRY_SCHEMA_VERSION
    assert manifest.run_id == "abc-run"
    assert manifest.provenance_digest == "dead"
    assert manifest.hardware is not None
    assert manifest.os is not None
    assert manifest.toolchain is not None
    assert manifest.repo is not None
    assert manifest.determinism is not None
    assert manifest.determinism.seed == 42
    assert manifest.determinism.batch_size == int(shared.latent.shape[0])
    assert manifest.cells_run == ["cell-a"]


def test_manifest_legacy_fields_mirror_structured_blocks(tmp_path):
    ws = _seed_workspace(tmp_path)
    shared = inputs.generate_shared_input(seed=1, resolution=64)
    manifest = collect_environment_manifest(
        seed=1,
        run_conditions="legacy-check",
        checkpoint_path=None,
        workspace=ws,
        shared_input=shared,
        run_id="r1",
    )
    # Legacy mirrors must be populated so v1 readers still see the same values
    # they used to read.
    assert manifest.chip_model == manifest.hardware.chip_brand
    assert manifest.seed == 1
    assert manifest.run_conditions == "legacy-check"


def test_manifest_writes_history_copy_per_run(tmp_path):
    ws = _seed_workspace(tmp_path)
    shared = inputs.generate_shared_input(seed=2, resolution=64)
    manifest = collect_environment_manifest(
        seed=2,
        run_conditions="t",
        workspace=ws,
        shared_input=shared,
        run_id="run-X",
    )
    latest = ws.results_data_dir / "environment.json"
    history = ws.results_data_dir / "environments"
    write_environment_manifest(manifest, latest, history_dir=history)
    assert latest.exists()
    assert (history / "run-X.json").exists()
    assert json.loads((history / "run-X.json").read_text())["run_id"] == "run-X"


def test_manifest_serializes_to_json_without_loss(tmp_path):
    ws = _seed_workspace(tmp_path)
    shared = inputs.generate_shared_input(seed=99, resolution=64)
    manifest = collect_environment_manifest(
        seed=99,
        run_conditions="json-roundtrip",
        workspace=ws,
        shared_input=shared,
        run_id="rid",
    )
    payload = json.dumps(asdict(manifest), default=str)
    parsed = json.loads(payload)
    assert parsed["schema_version"] == TELEMETRY_SCHEMA_VERSION
    assert parsed["determinism"]["seed"] == 99
    assert "host_id_hash" in parsed["hardware"]


# ---------------------------------------------------------------------------
# tables carry captions when a manifest is supplied
# ---------------------------------------------------------------------------


def _record(cell_id: str) -> BenchmarkRecord:
    return BenchmarkRecord(
        run_id="rid",
        cell_id=cell_id,
        backend="mlx",
        requested_compute_unit="GPU",
        realized_compute_unit="GPU",
        attention="NATIVE",
        precision="fp16",
        resolution=64,
        status="ok",
        latency_ms_median=10.0,
        latency_ms_iqr=1.0,
        gpu_power_w=None,
        ane_power_w=None,
        energy_per_unet_step_j=None,
        estimated_energy_per_50_step_image_j=None,
        mse=None,
        cosine=None,
        numerically_divergent=None,
        on_disk_size_bytes=None,
        weight_only_size_bytes=None,
        effective_bits_per_parameter=None,
        compute_precision="fp16",
        graph_capture_s=None,
        convert_s=None,
        first_load_compile_s=None,
        failure_reason=None,
    )


def test_table_caption_includes_chip_and_digest(tmp_path):
    ws = _seed_workspace(tmp_path)
    shared = inputs.generate_shared_input(seed=1, resolution=64)
    manifest = collect_environment_manifest(
        seed=1,
        run_conditions="cap",
        workspace=ws,
        shared_input=shared,
        run_id="r-cap",
        provenance_digest="d" * 40,
    )
    write_summary_tables([_record("a")], ws.results_tables_dir, manifest=manifest)
    latency = (ws.results_tables_dir / "latency.md").read_text()
    assert latency.startswith("<!-- sdbench")
    assert manifest.hardware.chip_brand in latency
    assert "provenance dddddddddddd" in latency
    assert (ws.results_tables_dir / "environment.md").exists()


# ---------------------------------------------------------------------------
# Contributor report bundle
# ---------------------------------------------------------------------------


def _seed_report_run(tmp_path) -> Workspace:
    ws = _seed_workspace(tmp_path)
    shared = inputs.generate_shared_input(seed=5, resolution=64)
    manifest = collect_environment_manifest(
        seed=5,
        run_conditions="full",
        workspace=ws,
        shared_input=shared,
        shared_input_path=None,
        run_id="run-z",
        provenance_digest="abc123" + "0" * 58,
        cells_run=["cell-a"],
    )
    write_environment_manifest(manifest, ws.results_data_dir / "environment.json")
    digests = inputs.digest_shared_input(shared)
    from dataclasses import replace as _replace

    record = _replace(
        _record("cell-a"),
        provenance_digest=manifest.provenance_digest,
        host_id_hash=manifest.hardware.host_id_hash,
        latent_input_sha256=digests["latent"],
        text_embedding_input_sha256=digests["text_embedding"],
    )
    write_jsonl([record], ws.results_data_dir / "results.jsonl")
    write_summary_tables([record], ws.results_tables_dir, manifest=manifest)
    (ws.results_raw_dir / "run-z-powermetrics.plist").write_bytes(b"<plist/>")
    return ws


def test_build_report_creates_bundle_directory_and_zip(tmp_path):
    ws = _seed_report_run(tmp_path)
    bundle = build_report(ws, run_id="run-z", zip_bundle=True)
    assert bundle.suffix == ".zip"
    dir_bundle = ws.results_dir / "reports" / "run-z"
    assert (dir_bundle / "manifest.json").exists()
    assert (dir_bundle / "results.jsonl").exists()
    assert (dir_bundle / "tables").is_dir()
    assert (dir_bundle / "raw").is_dir()
    assert (dir_bundle / "matrix.yaml").exists()
    assert (dir_bundle / "README.md").exists()
    assert (dir_bundle / "SCHEMA.md").exists()
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        assert any(n.endswith("manifest.json") for n in names)


def test_anonymize_strips_pii_but_keeps_reproducibility_fields(tmp_path):
    ws = _seed_report_run(tmp_path)
    bundle = build_report(ws, run_id="run-z", zip_bundle=False, anonymize=True, salt="mysalt")
    manifest = json.loads((bundle / "manifest.json").read_text())
    # PII stripped
    assert manifest["repo"]["upstream_url"] is None
    assert manifest["repo"]["dirty_files"] == []
    assert manifest.get("checkpoint_path") in (None,)
    # host_id_hash re-hashed (still 16 hex, but different from original)
    assert len(manifest["hardware"]["host_id_hash"]) == 16
    # Reproducibility fields preserved
    assert manifest["determinism"]["seed"] == 5
    assert manifest["determinism"]["latent_sha256"]
    assert manifest["determinism"]["text_embedding_sha256"]
    assert manifest["schema_version"] == TELEMETRY_SCHEMA_VERSION


def test_anonymize_without_salt_raises(tmp_path):
    ws = _seed_report_run(tmp_path)
    try:
        build_report(ws, run_id="run-z", zip_bundle=False, anonymize=True, salt=None)
    except ValueError as exc:
        assert "salt" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_validate_report_passes_on_clean_bundle(tmp_path):
    ws = _seed_report_run(tmp_path)
    bundle = build_report(ws, run_id="run-z", zip_bundle=False)
    result = validate_report(bundle)
    assert result.ok, result.issues


def test_validate_report_flags_schema_higher_than_supported(tmp_path):
    ws = _seed_report_run(tmp_path)
    bundle = build_report(ws, run_id="run-z", zip_bundle=False)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = TELEMETRY_SCHEMA_VERSION + 999
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    result = validate_report(bundle)
    assert result.ok is False
    assert not result.schema_ok


def test_validate_report_flags_drifted_latent_sha(tmp_path):
    ws = _seed_report_run(tmp_path)
    bundle = build_report(ws, run_id="run-z", zip_bundle=False)
    # Tamper one record's latent SHA so it no longer matches the manifest.
    results = (bundle / "results.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in results if line.strip()]
    rows[0]["latent_input_sha256"] = "0" * 64
    (bundle / "results.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8"
    )
    result = validate_report(bundle)
    assert result.ok is False
    assert not result.latent_consistent
