from dataclasses import replace

from sdbench.config import load_benchmark_config
from sdbench.provenance import Fingerprint, verify_results
from sdbench.results import BenchmarkRecord
from sdbench.tui.app import assess_state
from sdbench.tui.workspace import Workspace


def _fp(digest_seed: str = "abc") -> Fingerprint:
    return Fingerprint(
        checkpoint_sha256=digest_seed, tool_version="0.1.0", chip="Apple M2 Pro",
        harness_deps={"torch": "2.6.0"}, convert_ct8_deps={}, convert_ct9_deps={},
    )


def _record(cell_id: str, digest: str | None) -> BenchmarkRecord:
    base = BenchmarkRecord(
        run_id="r", cell_id=cell_id, backend="mlx", requested_compute_unit="GPU",
        realized_compute_unit="GPU", attention="NATIVE", precision="fp16", resolution=512,
        status="ok", latency_ms_median=1.0, latency_ms_iqr=0.1, gpu_power_w=None, ane_power_w=None,
        energy_per_unet_step_j=None, estimated_energy_per_50_step_image_j=None, mse=None, cosine=None,
        numerically_divergent=None, on_disk_size_bytes=None, weight_only_size_bytes=None,
        effective_bits_per_parameter=None, compute_precision="fp16", graph_capture_s=None,
        convert_s=None, first_load_compile_s=None, failure_reason=None,
    )
    return replace(base, provenance_digest=digest)


# ----- provenance verification -----

def test_verify_results_ok_when_single_matching_digest():
    fp = _fp()
    report = verify_results([_record("a", fp.digest), _record("b", fp.digest)], fp)
    assert report.ok and report.consistent and report.matches_current


def test_verify_results_flags_mixed_digests():
    fp = _fp()
    report = verify_results([_record("a", fp.digest), _record("b", "different")], fp)
    assert report.consistent is False
    assert report.ok is False


def test_verify_results_flags_foreign_environment():
    fp = _fp()
    report = verify_results([_record("a", "made-elsewhere")], fp)
    assert report.consistent is True  # one digest...
    assert report.matches_current is False  # ...but not this environment's


# ----- workspace state assessment -----

def _matrix(tmp_path, checkpoint: str) -> str:
    path = tmp_path / "matrix.yaml"
    path.write_text(
        f"""
checkpoint: {checkpoint}
seed: 0
warmup: 1
iterations: 10
resolution_default: 512
equivalence: {{ mse_max: 1.0e-3, cosine_min: 0.999 }}
power: {{ interval_ms: 100, baseline_seconds: 2 }}
thermal: {{ abort_on_throttle: false }}
cells:
  - {{ id: apple-ane, backend: apple_coreml, compute_unit: CPU_AND_NE, attention: SPLIT_EINSUM_V2, precision: fp16, resolution: 512 }}
  - {{ id: mlx, backend: mlx, compute_unit: GPU, attention: NATIVE, precision: fp16, resolution: 512 }}
""",
        encoding="utf-8",
    )
    return str(path)


def test_assess_state_empty_workspace(tmp_path):
    ws = Workspace.resolve(tmp_path)
    cfg = load_benchmark_config(_matrix(tmp_path, str(tmp_path / "missing.safetensors")))
    state = assess_state(ws, cfg)
    assert state.checkpoint_present is False
    assert state.artifacts_total == 1 and state.artifacts_present == 0  # only the apple build counts
    assert state.has_runplan is False and state.has_results is False
    assert state.has_report is False


def test_assess_state_detects_report_bundle_zip(tmp_path):
    ws = Workspace.resolve(tmp_path)
    cfg = load_benchmark_config(_matrix(tmp_path, str(tmp_path / "missing.safetensors")))
    reports = ws.results_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "run-abc.zip").write_bytes(b"PK")
    assert assess_state(ws, cfg).has_report is True


def test_main_menu_includes_report_entry():
    from sdbench.tui.app import MENU

    keys = [key for key, _ in MENU]
    assert "report" in keys
    # report appears AFTER run (post-benchmark step) and BEFORE cleanup (terminal action).
    assert keys.index("report") > keys.index("run")
    assert keys.index("report") < keys.index("cleanup")


def test_assess_state_detects_checkpoint_and_artifact(tmp_path):
    ckpt = tmp_path / "sd15.safetensors"
    ckpt.write_bytes(b"weights")
    ws = Workspace.resolve(tmp_path)
    cfg = load_benchmark_config(_matrix(tmp_path, str(ckpt)))

    artifact = ws.artifacts_dir / "apple_coreml" / "split_einsum_v2_ane" / "Stable_Diffusion_version_local_sd15_unet.mlmodelc"
    artifact.mkdir(parents=True)

    state = assess_state(ws, cfg)
    assert state.checkpoint_present is True and state.checkpoint == ckpt
    assert state.artifacts_present == 1 and state.artifacts_total == 1
