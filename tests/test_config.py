import os

import pytest

from sdbench.config import load_benchmark_config


def test_loads_matrix_and_expands_checkpoint_env_var(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("SD15_CHECKPOINT", str(checkpoint))
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        """
checkpoint: ${SD15_CHECKPOINT}
seed: 123
iterations: 11
warmup: 1
thermal:
  throttle_policy: abort
equivalence:
  mse_max: 1.0e-3
  cosine_min: 0.999
power:
  interval_ms: 100
  baseline_seconds: 2
cells:
  - { id: mlx-fp16, backend: mlx, compute_unit: GPU, attention: NATIVE, precision: fp16, resolution: 512 }
""",
        encoding="utf-8",
    )

    cfg = load_benchmark_config(config_path)

    assert cfg.checkpoint == checkpoint
    assert cfg.seed == 123
    assert cfg.iterations == 11
    assert cfg.thermal.throttle_policy == "abort"
    assert cfg.cells[0].id == "mlx-fp16"


def test_loads_rich_matrix_schema_and_filters_enabled_cells(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("SD15_CHECKPOINT", str(checkpoint))
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        """
checkpoint: ${SD15_CHECKPOINT}
seed: 0
iterations: 10
warmup: 1
resolution_default: 512
thermal: { abort_on_throttle: true }
equivalence:
  reference: { backend: diffusers_mps, device: cpu, precision: fp32 }
  mse_max: 1.0e-3
  cosine_min: 0.999
power:
  interval_ms: 100
  baseline_seconds: 2
  samplers: [cpu_power, gpu_power, ane_power]
cells:
  - id: diffusers-mps-fp16
    label: "diffusers MPS fp16"
    backend: diffusers_mps
    compute_unit: MPS
    attention: NATIVE
    precision: fp16
    enabled: true
  - id: mlx-q8
    backend: mlx
    compute_unit: GPU
    attention: NATIVE
    precision: w8
    enabled: false
    requires: { mlx_quant: true }
    notes: "Optional."
""",
        encoding="utf-8",
    )

    cfg = load_benchmark_config(config_path)

    assert cfg.equivalence.reference == {"backend": "diffusers_mps", "device": "cpu", "precision": "fp32"}
    assert cfg.power.samplers == ["cpu_power", "gpu_power", "ane_power"]
    assert cfg.thermal.abort_on_throttle is True
    assert cfg.cells[0].resolution == 512
    assert cfg.cells[0].label == "diffusers MPS fp16"
    assert cfg.cells[1].requires == {"mlx_quant": True}
    assert [cell.id for cell in cfg.enabled_cells()] == ["diffusers-mps-fp16"]
    assert cfg.select_cell_by_id("mlx-q8").enabled is False


def test_rejects_less_than_ten_timed_iterations(tmp_path):
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        """
checkpoint: /tmp/model.safetensors
seed: 0
iterations: 9
warmup: 1
equivalence: { mse_max: 1.0e-3, cosine_min: 0.999 }
power: { interval_ms: 100, baseline_seconds: 2 }
cells:
  - { id: bad, backend: mlx, compute_unit: GPU, attention: NATIVE, precision: fp16, resolution: 512 }
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least 10"):
        load_benchmark_config(config_path)


def test_errors_when_checkpoint_env_var_is_missing(tmp_path):
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        """
checkpoint: ${MISSING_SD15_CHECKPOINT}
seed: 0
iterations: 10
warmup: 1
equivalence: { mse_max: 1.0e-3, cosine_min: 0.999 }
power: { interval_ms: 100, baseline_seconds: 2 }
cells:
  - { id: bad, backend: mlx, compute_unit: GPU, attention: NATIVE, precision: fp16, resolution: 512 }
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="MISSING_SD15_CHECKPOINT"):
        load_benchmark_config(config_path)


def test_loads_checkpoint_from_dotenv_next_to_config(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")
    (tmp_path / ".env").write_text(f"SD15_CHECKPOINT={checkpoint}\n", encoding="utf-8")
    monkeypatch.delenv("SD15_CHECKPOINT", raising=False)
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        """
checkpoint: ${SD15_CHECKPOINT}
seed: 0
iterations: 10
warmup: 1
equivalence: { mse_max: 1.0e-3, cosine_min: 0.999 }
power: { interval_ms: 100, baseline_seconds: 2 }
cells:
  - { id: mlx-fp16, backend: mlx, compute_unit: GPU, attention: NATIVE, precision: fp16, resolution: 512 }
""",
        encoding="utf-8",
    )

    cfg = load_benchmark_config(config_path)

    assert cfg.checkpoint == checkpoint


def test_loads_checkpoint_from_dotenv_in_working_directory(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (tmp_path / ".env").write_text(f"SD15_CHECKPOINT={checkpoint}\n", encoding="utf-8")
    monkeypatch.delenv("SD15_CHECKPOINT", raising=False)
    monkeypatch.chdir(tmp_path)
    config_path = config_dir / "matrix.yaml"
    config_path.write_text(
        """
checkpoint: ${SD15_CHECKPOINT}
seed: 0
iterations: 10
warmup: 1
equivalence: { mse_max: 1.0e-3, cosine_min: 0.999 }
power: { interval_ms: 100, baseline_seconds: 2 }
cells:
  - { id: mlx-fp16, backend: mlx, compute_unit: GPU, attention: NATIVE, precision: fp16, resolution: 512 }
""",
        encoding="utf-8",
    )

    cfg = load_benchmark_config(config_path)

    assert cfg.checkpoint == checkpoint


def test_selects_single_cell_by_requested_fields(tmp_path):
    monkeypatch = pytest.MonkeyPatch()
    checkpoint = tmp_path / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("SD15_CHECKPOINT", str(checkpoint))
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        """
checkpoint: ${SD15_CHECKPOINT}
seed: 0
iterations: 10
warmup: 1
equivalence: { mse_max: 1.0e-3, cosine_min: 0.999 }
power: { interval_ms: 100, baseline_seconds: 2 }
cells:
  - { id: mlx-fp16, backend: mlx, compute_unit: GPU, attention: NATIVE, precision: fp16, resolution: 512 }
  - { id: diffusers-fp16, backend: diffusers_mps, compute_unit: MPS, attention: NATIVE, precision: fp16, resolution: 512 }
""",
        encoding="utf-8",
    )

    cfg = load_benchmark_config(config_path)
    selected = cfg.select_cell(
        backend="diffusers_mps",
        compute_unit="MPS",
        attention="NATIVE",
        precision="fp16",
        resolution=512,
    )

    assert selected.id == "diffusers-fp16"
    monkeypatch.undo()
