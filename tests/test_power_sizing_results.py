from pathlib import Path

import numpy as np
import pytest

from sdbench.power import PowerSample, summarize_power
from sdbench.results import BenchmarkRecord, load_jsonl, write_jsonl, write_summary_tables
from safetensors.torch import save_file
import torch

from sdbench.sizing import artifact_size_bytes, compute_quantization_efficiency, effective_bits_per_parameter, safetensors_parameter_count, safetensors_weight_size


def test_power_summary_baseline_subtracts_and_reports_unet_step_energy():
    samples = [
        PowerSample(timestamp_s=0.0, cpu_w=1.0, gpu_w=2.0, ane_w=0.5),
        PowerSample(timestamp_s=0.5, cpu_w=1.0, gpu_w=2.0, ane_w=0.5),
        PowerSample(timestamp_s=1.0, cpu_w=1.0, gpu_w=7.0, ane_w=3.5),
        PowerSample(timestamp_s=1.5, cpu_w=1.0, gpu_w=9.0, ane_w=4.5),
        PowerSample(timestamp_s=2.0, cpu_w=1.0, gpu_w=2.0, ane_w=0.5),
        PowerSample(timestamp_s=2.5, cpu_w=1.0, gpu_w=2.0, ane_w=0.5),
    ]

    summary = summarize_power(samples, active_start_s=1.0, active_end_s=2.0, timed_iterations=10)

    assert summary.gpu_power_w == 6.0
    assert summary.ane_power_w == 3.5
    assert summary.energy_per_unet_step_j == 0.95
    assert summary.estimated_energy_per_50_step_image_j == 47.5


def test_sizing_reports_directory_size_and_effective_bits(tmp_path):
    artifact = tmp_path / "model.mlmodelc"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"a" * 32)
    (artifact / "metadata.json").write_bytes(b"b" * 8)

    assert artifact_size_bytes(artifact) == 40
    assert effective_bits_per_parameter(weight_bytes=32, parameter_count=64) == 4.0


def test_safetensors_weight_size_filters_unet_prefix(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "model.diffusion_model.block.weight": torch.zeros((2, 4), dtype=torch.float16),
            "cond_stage_model.text.weight": torch.zeros((10, 10), dtype=torch.float32),
        },
        path,
    )

    size = safetensors_weight_size(path, ("model.diffusion_model.",), compute_precision="fp16")

    assert size.on_disk_size_bytes == artifact_size_bytes(path)
    assert size.weight_only_size_bytes == 16
    assert size.effective_bits_per_parameter == 16.0
    assert size.compute_precision == "fp16"


def test_safetensors_weight_size_normalises_fp32_storage_to_fp16_runtime(tmp_path):
    """SD 1.5 ships fp32 on Hugging Face; diffusers/MLX cast to fp16 at load.
    The reported weight footprint must reflect the in-memory runtime size so
    the size table compares fp16-vs-fp16, not fp32-storage-vs-fp16-mlpackage."""
    path = tmp_path / "fp32_checkpoint.safetensors"
    save_file(
        {"model.diffusion_model.block.weight": torch.zeros((4, 4), dtype=torch.float32)},
        path,
    )

    size = safetensors_weight_size(path, ("model.diffusion_model.",), compute_precision="fp16")

    # 16 parameters × 2 bytes (fp16) = 32, not 64 (fp32 storage).
    assert size.weight_only_size_bytes == 32
    assert size.effective_bits_per_parameter == 16.0
    # on_disk still reflects the real fp32 file: 16 × 4 bytes plus safetensors header.
    assert size.on_disk_size_bytes >= 64


def test_safetensors_parameter_count_filters_prefix(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "model.diffusion_model.block.weight": torch.zeros((2, 4), dtype=torch.float16),
            "cond_stage_model.text.weight": torch.zeros((10, 10), dtype=torch.float32),
        },
        path,
    )

    assert safetensors_parameter_count(path, ("model.diffusion_model.",)) == 8


def test_quantization_efficiency_uses_fp16_backend_baseline():
    efficiency = compute_quantization_efficiency(
        fp16_size_bytes=100,
        quant_size_bytes=25,
        fp16_latency_ms=10,
        quant_latency_ms=8,
        fp16_mse=0.0,
        quant_mse=0.002,
        fp16_cosine=1.0,
        quant_cosine=0.998,
    )

    assert efficiency.size_reduction_ratio == 0.75
    assert efficiency.latency_change_ratio == -0.2
    assert efficiency.mse_delta == 0.002
    assert efficiency.cosine_delta == pytest.approx(-0.002)


def test_summary_tables_use_unet_step_energy_name(tmp_path):
    record = BenchmarkRecord(
        run_id="run",
        cell_id="mlx-fp16",
        backend="mlx",
        requested_compute_unit="GPU",
        realized_compute_unit="GPU",
        attention="NATIVE",
        precision="fp16",
        resolution=512,
        status="ok",
        latency_ms_median=10.0,
        latency_ms_iqr=1.0,
        gpu_power_w=6.0,
        ane_power_w=0.0,
        energy_per_unet_step_j=0.06,
        estimated_energy_per_50_step_image_j=3.0,
        mse=0.0,
        cosine=1.0,
        numerically_divergent=False,
        on_disk_size_bytes=100,
        weight_only_size_bytes=80,
        effective_bits_per_parameter=16.0,
        compute_precision="fp16",
        graph_capture_s=None,
        convert_s=None,
        first_load_compile_s=None,
        failure_reason=None,
    )

    write_summary_tables([record], tmp_path)

    power_table = (tmp_path / "power_energy.md").read_text(encoding="utf-8")
    assert "Energy / UNet step (J)" in power_table
    assert "Estimated energy / 50-step image (J)" in power_table


def test_equivalence_table_preserves_subepsilon_precision(tmp_path):
    """Reader must see real drift between fp16 backend and fp32 CPU reference;
    rounding to four decimals would hide a cosine of 0.9999996 as 1.0000."""
    flagged = BenchmarkRecord(
        run_id="run",
        cell_id="ane-cell",
        backend="apple_coreml",
        requested_compute_unit="CPU_AND_NE",
        realized_compute_unit="CPU_AND_NE",
        attention="SPLIT_EINSUM_V2",
        precision="fp16",
        resolution=512,
        status="ok",
        latency_ms_median=400.0,
        latency_ms_iqr=0.3,
        gpu_power_w=None, ane_power_w=None,
        energy_per_unet_step_j=None, estimated_energy_per_50_step_image_j=None,
        mse=5.65e-3, cosine=0.99689, numerically_divergent=True,
        on_disk_size_bytes=None, weight_only_size_bytes=None,
        effective_bits_per_parameter=None, compute_precision="fp16",
        graph_capture_s=None, convert_s=None, first_load_compile_s=None,
        failure_reason=None,
    )
    clean = BenchmarkRecord(
        run_id="run",
        cell_id="gpu-cell",
        backend="apple_coreml",
        requested_compute_unit="CPU_AND_GPU",
        realized_compute_unit="CPU_AND_GPU",
        attention="ORIGINAL",
        precision="fp16",
        resolution=512,
        status="ok",
        latency_ms_median=440.0,
        latency_ms_iqr=1.0,
        gpu_power_w=None, ane_power_w=None,
        energy_per_unet_step_j=None, estimated_energy_per_50_step_image_j=None,
        mse=6.13e-7, cosine=0.9999996, numerically_divergent=False,
        on_disk_size_bytes=None, weight_only_size_bytes=None,
        effective_bits_per_parameter=None, compute_precision="fp16",
        graph_capture_s=None, convert_s=None, first_load_compile_s=None,
        failure_reason=None,
    )

    write_summary_tables([flagged, clean], tmp_path)

    table = (tmp_path / "equivalence.md").read_text(encoding="utf-8")
    assert "| MSE | Cosine | Flagged |" in table
    assert "5.650e-03" in table and "0.9968900" in table and "| yes |" in table
    # Critical: a cosine of 0.9999996 must NOT round to 1.0 in the table.
    assert "0.9999996" in table
    assert "| no |" in table


def test_results_jsonl_round_trips_records(tmp_path):
    record = BenchmarkRecord(
        run_id="run",
        cell_id="mlx-fp16",
        backend="mlx",
        requested_compute_unit="GPU",
        realized_compute_unit="GPU",
        attention="NATIVE",
        precision="fp16",
        resolution=512,
        status="ok",
        latency_ms_median=10.0,
        latency_ms_iqr=1.0,
        gpu_power_w=6.0,
        ane_power_w=0.0,
        energy_per_unet_step_j=0.06,
        estimated_energy_per_50_step_image_j=3.0,
        mse=0.0,
        cosine=1.0,
        numerically_divergent=False,
        on_disk_size_bytes=100,
        weight_only_size_bytes=80,
        effective_bits_per_parameter=16.0,
        compute_precision="fp16",
        graph_capture_s=None,
        convert_s=None,
        first_load_compile_s=None,
        failure_reason=None,
    )
    path = tmp_path / "results.jsonl"

    write_jsonl([record], path)
    loaded = load_jsonl(path)

    assert loaded == [record]


def test_results_jsonl_writes_nonfinite_floats_as_null(tmp_path):
    record = BenchmarkRecord(
        run_id="run",
        cell_id="bad",
        backend="coreml_diffusion",
        requested_compute_unit="CPU_AND_GPU",
        realized_compute_unit="CPU_AND_GPU",
        attention="ORIGINAL",
        precision="fp16",
        resolution=512,
        status="ok",
        latency_ms_median=10.0,
        latency_ms_iqr=1.0,
        gpu_power_w=0.0,
        ane_power_w=0.0,
        energy_per_unet_step_j=0.0,
        estimated_energy_per_50_step_image_j=0.0,
        mse=float("nan"),
        cosine=float("nan"),
        numerically_divergent=True,
        on_disk_size_bytes=100,
        weight_only_size_bytes=80,
        effective_bits_per_parameter=16.0,
        compute_precision="fp16",
        graph_capture_s=None,
        convert_s=None,
        first_load_compile_s=None,
        failure_reason=None,
    )
    path = tmp_path / "results.jsonl"

    write_jsonl([record], path)

    assert '"mse": null' in path.read_text(encoding="utf-8")
