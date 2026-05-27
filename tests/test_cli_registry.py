import numpy as np
from typer.testing import CliRunner

from sdbench.backends.apple_coreml import AppleCoreMLAdapter
from sdbench.backends.coreml_diffusion import CoreMLDiffusionAdapter
from sdbench.backends.registry import build_default_adapters
from sdbench.backends.diffusers_mps import DiffusersMpsAdapter
from sdbench.cli import app


def test_default_registry_exposes_all_spec_backends():
    adapters = build_default_adapters(checkpoint_path="/tmp/model.safetensors")

    assert set(adapters) == {"apple_coreml", "coreml_diffusion", "diffusers_mps", "mlx"}
    assert isinstance(adapters["apple_coreml"], AppleCoreMLAdapter)
    assert isinstance(adapters["coreml_diffusion"], CoreMLDiffusionAdapter)
    assert isinstance(adapters["diffusers_mps"], DiffusersMpsAdapter)


def test_cli_help_renders_without_traceback():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "prepare-input" in result.output


def test_prepare_input_command_writes_shared_input(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("SD15_CHECKPOINT", str(checkpoint))
    config_path = tmp_path / "matrix.yaml"
    output_path = tmp_path / "shared_input.npz"
    config_path.write_text(
        """
checkpoint: ${SD15_CHECKPOINT}
seed: 9
iterations: 10
warmup: 1
thermal: { throttle_policy: abort }
equivalence: { mse_max: 1.0e-3, cosine_min: 0.999 }
power: { interval_ms: 100, baseline_seconds: 2 }
cells:
  - { id: mlx-fp16, backend: mlx, compute_unit: GPU, attention: NATIVE, precision: fp16, resolution: 512 }
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["prepare-input", "--config", str(config_path), "--output", str(output_path)],
    )

    assert result.exit_code == 0
    data = np.load(output_path)
    assert data["latent"].shape == (2, 4, 64, 64)


def test_run_cell_accepts_cell_id(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")
    shared_input = tmp_path / "shared.npz"
    np.savez_compressed(
        shared_input,
        latent=np.zeros((1, 1), dtype=np.float32),
        timestep=np.array(1, dtype=np.int64),
        text_embedding=np.zeros((1, 1), dtype=np.float32),
    )
    monkeypatch.setenv("SD15_CHECKPOINT", str(checkpoint))
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        """
checkpoint: ${SD15_CHECKPOINT}
seed: 0
iterations: 10
warmup: 1
thermal: { throttle_policy: abort }
equivalence: { mse_max: 1.0e-3, cosine_min: 0.999 }
power: { interval_ms: 100, baseline_seconds: 2 }
cells:
  - { id: missing-apple, backend: apple_coreml, compute_unit: CPU_AND_GPU, attention: ORIGINAL, precision: fp16, resolution: 512 }
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "run-cell",
            "--cell",
            "missing-apple",
            "--config",
            str(config_path),
            "--shared-input",
            str(shared_input),
            "--results-dir",
            str(tmp_path / "results"),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "results" / "data" / "missing-apple.jsonl").exists()
