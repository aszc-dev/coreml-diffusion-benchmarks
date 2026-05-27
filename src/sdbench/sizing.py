from dataclasses import dataclass
from pathlib import Path
from math import prod


@dataclass(frozen=True)
class QuantizationEfficiency:
    size_reduction_ratio: float
    latency_change_ratio: float
    mse_delta: float
    cosine_delta: float


@dataclass(frozen=True)
class ModelSize:
    on_disk_size_bytes: int
    weight_only_size_bytes: int
    effective_bits_per_parameter: float
    compute_precision: str


def artifact_size_bytes(path: str | Path) -> int:
    artifact = Path(path)
    if artifact.is_file():
        return artifact.stat().st_size
    return sum(item.stat().st_size for item in artifact.rglob("*") if item.is_file())


def effective_bits_per_parameter(weight_bytes: int, parameter_count: int) -> float:
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    return (weight_bytes * 8.0) / parameter_count


def safetensors_weight_size(
    path: str | Path,
    key_prefixes: tuple[str, ...],
    compute_precision: str,
) -> ModelSize:
    from safetensors import safe_open

    weight_bytes = 0
    parameter_count = 0
    with safe_open(Path(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith(key_prefixes):
                continue
            tensor_slice = handle.get_slice(key)
            shape = tensor_slice.get_shape()
            dtype = tensor_slice.get_dtype()
            numel = prod(shape)
            parameter_count += numel
            weight_bytes += numel * _dtype_bytes(dtype)
    return ModelSize(
        on_disk_size_bytes=artifact_size_bytes(path),
        weight_only_size_bytes=weight_bytes,
        effective_bits_per_parameter=effective_bits_per_parameter(weight_bytes, parameter_count),
        compute_precision=compute_precision,
    )


def safetensors_parameter_count(path: str | Path, key_prefixes: tuple[str, ...]) -> int:
    from safetensors import safe_open

    parameter_count = 0
    with safe_open(Path(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(key_prefixes):
                parameter_count += prod(handle.get_slice(key).get_shape())
    return parameter_count


def _dtype_bytes(dtype: str) -> int:
    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E5M2": 1,
        "F8_E4M3": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    if dtype not in sizes:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")
    return sizes[dtype]


def compute_quantization_efficiency(
    fp16_size_bytes: int,
    quant_size_bytes: int,
    fp16_latency_ms: float,
    quant_latency_ms: float,
    fp16_mse: float,
    quant_mse: float,
    fp16_cosine: float,
    quant_cosine: float,
) -> QuantizationEfficiency:
    return QuantizationEfficiency(
        size_reduction_ratio=(fp16_size_bytes - quant_size_bytes) / fp16_size_bytes,
        latency_change_ratio=(quant_latency_ms - fp16_latency_ms) / fp16_latency_ms,
        mse_delta=quant_mse - fp16_mse,
        cosine_delta=quant_cosine - fp16_cosine,
    )
