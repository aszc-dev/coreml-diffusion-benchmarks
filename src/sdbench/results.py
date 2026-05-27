import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRecord:
    run_id: str
    cell_id: str
    backend: str
    requested_compute_unit: str
    realized_compute_unit: str | None
    attention: str
    precision: str
    resolution: int
    status: str
    latency_ms_median: float | None
    latency_ms_iqr: float | None
    gpu_power_w: float | None
    ane_power_w: float | None
    energy_per_unet_step_j: float | None
    estimated_energy_per_50_step_image_j: float | None
    mse: float | None
    cosine: float | None
    numerically_divergent: bool | None
    on_disk_size_bytes: int | None
    weight_only_size_bytes: int | None
    effective_bits_per_parameter: float | None
    compute_precision: str | None
    graph_capture_s: float | None
    convert_s: float | None
    first_load_compile_s: float | None
    failure_reason: str | None
    # Wall-clock (epoch) bounds of the timed window the power figures are aligned to (R6.3).
    # Defaulted so existing constructors and older JSONL records remain valid.
    active_window_start_s: float | None = None
    active_window_end_s: float | None = None


def write_jsonl(records: list[BenchmarkRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(asdict(record)), allow_nan=False, sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> list[BenchmarkRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(BenchmarkRecord(**json.loads(line)))
    return records


def write_summary_tables(records: list[BenchmarkRecord], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_table(
        out / "latency.md",
        ["Cell", "Backend", "Median latency (ms)", "IQR (ms)", "Status"],
        [
            [
                record.cell_id,
                record.backend,
                _fmt(record.latency_ms_median),
                _fmt(record.latency_ms_iqr),
                record.status,
            ]
            for record in records
        ],
    )
    _write_table(
        out / "power_energy.md",
        [
            "Cell",
            "GPU power (W)",
            "ANE power (W)",
            "Energy / UNet step (J)",
            "Estimated energy / 50-step image (J)",
        ],
        [
            [
                record.cell_id,
                _fmt(record.gpu_power_w),
                _fmt(record.ane_power_w),
                _fmt(record.energy_per_unet_step_j),
                _fmt(record.estimated_energy_per_50_step_image_j),
            ]
            for record in records
        ],
    )
    _write_table(
        out / "size_quantization.md",
        ["Cell", "On-disk size (bytes)", "Weight-only size (bytes)", "Bits / parameter", "Compute precision"],
        [
            [
                record.cell_id,
                _fmt(record.on_disk_size_bytes),
                _fmt(record.weight_only_size_bytes),
                _fmt(record.effective_bits_per_parameter),
                record.compute_precision or "N/A",
            ]
            for record in records
        ],
    )
    _write_table(
        out / "conversion_time.md",
        ["Cell", "Graph capture (s)", "Convert (s)", "First-load compile (s)"],
        [
            [
                record.cell_id,
                _fmt(record.graph_capture_s),
                _fmt(record.convert_s),
                _fmt(record.first_load_compile_s),
            ]
            for record in records
        ],
    )


def _write_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
