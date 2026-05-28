import hashlib

import pytest

from sdbench.config import CellConfig
from sdbench.tui.convert_orchestrator import (
    build_command,
    conversion_timings_by_cell,
    convert_all,
    is_cached,
    plan_conversions,
    write_sidecar,
)
from sdbench.tui.download import CheckpointSpec, resolve_checkpoint, verify_sha256
from sdbench.tui.preflight import load_footprint
from sdbench.tui.sizing_probe import measure_cell_footprint, write_footprint
from sdbench.tui.workspace import Workspace


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----- download -----

def test_verify_sha256(tmp_path):
    f = tmp_path / "w.safetensors"
    f.write_bytes(b"weights")
    assert verify_sha256(f, _sha(b"weights"))
    assert not verify_sha256(f, "0" * 64)


def test_resolve_checkpoint_explicit_match(tmp_path):
    ws = Workspace.resolve(tmp_path)
    f = tmp_path / "local.safetensors"
    f.write_bytes(b"weights")
    spec = CheckpointSpec(repo="r", filename="local.safetensors", sha256=_sha(b"weights"))
    assert resolve_checkpoint(ws, spec, explicit=f) == f


def test_resolve_checkpoint_explicit_mismatch_raises(tmp_path):
    ws = Workspace.resolve(tmp_path)
    f = tmp_path / "local.safetensors"
    f.write_bytes(b"weights")
    spec = CheckpointSpec(repo="r", filename="local.safetensors", sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA mismatch"):
        resolve_checkpoint(ws, spec, explicit=f)


def test_resolve_checkpoint_auto_download_then_verify(tmp_path):
    ws = Workspace.resolve(tmp_path)
    spec = CheckpointSpec(repo="r", filename="ckpt.safetensors", sha256=_sha(b"the-weights"))

    def fake_downloader(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"the-weights")
        return dest

    path = resolve_checkpoint(ws, spec, auto_download=True, downloader=fake_downloader)
    assert path == ws.cache_dir / "ckpt.safetensors"
    assert path.read_bytes() == b"the-weights"


def test_resolve_checkpoint_no_source_raises(tmp_path):
    ws = Workspace.resolve(tmp_path)
    spec = CheckpointSpec(repo="r", filename="ckpt.safetensors", sha256="0" * 64)
    with pytest.raises(FileNotFoundError, match="No verified checkpoint"):
        resolve_checkpoint(ws, spec, auto_download=False)


# ----- conversion orchestration -----

class _Cfg:
    def __init__(self, cells):
        self.cells = cells


def _coreml_cells():
    return [
        CellConfig(id="apple-ane", backend="apple_coreml", compute_unit="CPU_AND_NE", attention="SPLIT_EINSUM_V2", precision="fp16", resolution=512),
        CellConfig(id="ours-ane", backend="coreml_diffusion", compute_unit="CPU_AND_NE", attention="SPLIT_EINSUM_V2", precision="fp16", resolution=512),
        CellConfig(id="mlx", backend="mlx", compute_unit="GPU", attention="NATIVE", precision="fp16", resolution=512),
    ]


def test_plan_conversions_covers_coreml_only(tmp_path):
    ws = Workspace.resolve(tmp_path)
    builds = {b.backend for b in plan_conversions(ws, _Cfg(_coreml_cells()))}
    assert builds == {"apple_coreml", "coreml_diffusion"}  # mlx skipped


def test_plan_conversions_skips_cells_without_buildable_artifact(tmp_path):
    # a gated precision (w8a8) has no supported artifact path; it must be skipped, not crash
    ws = Workspace.resolve(tmp_path)
    cells = _coreml_cells() + [
        CellConfig(id="w8a8", backend="coreml_diffusion", compute_unit="CPU_AND_NE", attention="SPLIT_EINSUM_V2", precision="w8a8", resolution=512),
    ]
    ids = {cid for b in plan_conversions(ws, _Cfg(cells)) for cid in b.cell_ids}
    assert "w8a8" not in ids
    assert {"apple-ane", "ours-ane"} <= ids


def test_build_command_uses_isolated_env(tmp_path):
    ws = Workspace.resolve(tmp_path)
    apple_build = next(b for b in plan_conversions(ws, _Cfg(_coreml_cells())) if b.backend == "apple_coreml")
    cmd = build_command(apple_build, "/ckpt.safetensors")
    assert cmd[:4] == ["uv", "run", "--project", str(ws.root / "envs" / "apple-ct8")]
    assert "--checkpoint" in cmd and "/ckpt.safetensors" in cmd
    assert "--attention" in cmd and "SPLIT_EINSUM_V2" in cmd


def test_is_cached_respects_source_sha(tmp_path):
    ws = Workspace.resolve(tmp_path)
    build = next(b for b in plan_conversions(ws, _Cfg(_coreml_cells())) if b.backend == "coreml_diffusion")
    build.expected_artifact.parent.mkdir(parents=True, exist_ok=True)
    build.expected_artifact.mkdir()  # .mlmodelc is a directory

    assert is_cached(build, "sha-A") is False  # no sidecar yet
    write_sidecar(build, "sha-A")
    assert is_cached(build, "sha-A") is True
    assert is_cached(build, "sha-B") is False  # checkpoint changed -> stale


def test_convert_all_runs_missing_builds(tmp_path):
    ws = Workspace.resolve(tmp_path)
    cfg = _Cfg(_coreml_cells())
    calls = []
    ran = convert_all(ws, cfg, "/ckpt", "sha-A", runner=lambda cmd, check=False: calls.append(cmd))
    assert len(ran) == 2 and len(calls) == 2  # both coreml builds were missing


def test_stream_run_captures_lines():
    import sys

    from sdbench.tui.convert_orchestrator import _stream_run

    lines: list[str] = []
    _stream_run([sys.executable, "-c", "print('alpha'); print('beta')"], lines.append)
    assert lines == ["alpha", "beta"]


def test_stream_run_raises_on_failure():
    import subprocess
    import sys

    from sdbench.tui.convert_orchestrator import _stream_run

    with pytest.raises(subprocess.CalledProcessError):
        _stream_run([sys.executable, "-c", "import sys; sys.exit(3)"], lambda _: None)


def test_convert_all_invokes_progress_callbacks(tmp_path):
    ws = Workspace.resolve(tmp_path)
    cfg = _Cfg(_coreml_cells())
    builds = plan_conversions(ws, cfg)
    builds[0].expected_artifact.mkdir(parents=True, exist_ok=True)  # pre-cache the first build
    write_sidecar(builds[0], "sha")

    skipped, built = [], []
    convert_all(
        ws, cfg, "/ckpt", "sha",
        runner=lambda cmd, check=False: None,
        on_skip=lambda b: skipped.append(b),
        on_build=lambda b, i, t: built.append((b, i, t)),
    )
    assert len(skipped) == 1
    assert len(built) == 1 and built[0][2] == 1  # one build to run, total reported as 1


def test_convert_all_skips_valid_cache(tmp_path):
    ws = Workspace.resolve(tmp_path)
    cfg = _Cfg(_coreml_cells())
    for build in plan_conversions(ws, cfg):  # pre-seed exact artifacts + matching sidecars
        build.expected_artifact.mkdir(parents=True, exist_ok=True)
        write_sidecar(build, "sha-A")
    calls = []
    assert convert_all(ws, cfg, "/ckpt", "sha-A", runner=lambda cmd, check=False: calls.append(cmd)) == []
    assert calls == []


# ----- disk footprint -----

def test_conversion_timings_mapped_to_cells(tmp_path):
    ws = Workspace.resolve(tmp_path)
    cfg = _Cfg(_coreml_cells())
    apple = next(b for b in plan_conversions(ws, cfg) if b.backend == "apple_coreml")
    apple.output_dir.mkdir(parents=True, exist_ok=True)
    apple.timings_out.write_text(
        '{"graph_capture_s": 1.5, "convert_s": 42.0, "first_load_compile_s": 7.0}', encoding="utf-8"
    )

    mapped = conversion_timings_by_cell(ws, cfg)
    assert mapped["apple-ane"] == {"graph_capture_s": 1.5, "convert_s": 42.0, "first_load_compile_s": 7.0}
    assert "mlx" not in mapped  # non-CoreML cell has no conversion build


def test_measure_and_write_footprint_roundtrip(tmp_path):
    ws = Workspace.resolve(tmp_path)
    cfg = _Cfg(_coreml_cells())
    # create the apple ANE artifact directory with a known size
    apple_art = ws.artifacts_dir / "apple_coreml" / "split_einsum_v2_ane" / "Stable_Diffusion_version_local_sd15_unet.mlmodelc"
    apple_art.mkdir(parents=True)
    (apple_art / "weight.bin").write_bytes(b"x" * 4096)

    sizes = measure_cell_footprint(ws, cfg)
    assert sizes == {"apple-ane": 4096}  # only the existing artifact measured

    write_footprint(ws.disk_footprint_path, sizes)
    assert load_footprint(ws.disk_footprint_path) == {"apple-ane": 4096}
