import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SD 1.5 UNet with the coreml-diffusion coremltools 9 toolchain.")
    parser.add_argument("--checkpoint", required=True, help="Local SD 1.5 .safetensors checkpoint.")
    parser.add_argument("--output-dir", default="artifacts/coreml_diffusion", help="Directory for CoreML artifacts.")
    parser.add_argument("--attention", choices=["ORIGINAL", "SPLIT_EINSUM", "SPLIT_EINSUM_V2"], required=True)
    parser.add_argument("--compute-unit", choices=["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL"], required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--precision", choices=["fp16", "w8", "w6", "w4"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-label", default="local_sd15")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timings-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output_dir)
    quantize = _quantize_arg(args.precision)
    mlpackage_path = output_dir / f"{args.model_label}-{args.attention}-{args.precision}.mlpackage"
    timings_out = Path(args.timings_out) if args.timings_out else output_dir / f"{args.model_label}-{args.attention}-{args.compute_unit}-{args.precision}-conversion.json"

    plan = {
        "backend": "coreml_diffusion",
        "coremltools_major": 9,
        # `--dry-run` plan still shows the absolute paths since this is purely
        # a local debugging dump; the committed sidecar drops them.
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "mlpackage_path": str(mlpackage_path),
        "attention": args.attention,
        "compute_unit": args.compute_unit,
        "resolution": args.resolution,
        "precision": args.precision,
        "quantize": quantize,
        "batch_size": args.batch_size,
        "compile": args.compile,
        "timings_out": str(timings_out),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)

    timings = convert_unet(args, checkpoint, mlpackage_path, quantize)
    timings_out.parent.mkdir(parents=True, exist_ok=True)
    timings_out.write_text(json.dumps(timings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote conversion timings to {timings_out}")


def convert_unet(args: argparse.Namespace, checkpoint: Path, mlpackage_path: Path, quantize: str) -> dict:
    import torch
    import coreml_diffusion.convert as team
    from coreml_diffusion.model_version import ModelVersion

    timings: dict[str, float | str | None] = {
        "backend": "coreml_diffusion",
        "coremltools_major": 9,
        # NOTE: full checkpoint path intentionally omitted from this sidecar
        # ($HOME leak). The checkpoint SHA-256 lives in .source.json next to
        # the artifact and is the canonical identity for reproducibility.
        "attention": args.attention,
        "compute_unit": args.compute_unit,
        "resolution": args.resolution,
        "precision": args.precision,
        "quantize": quantize,
        "batch_size": args.batch_size,
        "graph_capture_s": None,
        "convert_s": None,
        "first_load_compile_s": None,
        "mlpackage_path": _relativize(mlpackage_path),
        "mlmodelc_path": None,
        "status": "started",
    }

    original_trace = torch.jit.trace
    original_convert = team.convert_to_coreml

    def timed_trace(*trace_args, **trace_kwargs):
        start = time.monotonic()
        try:
            return original_trace(*trace_args, **trace_kwargs)
        finally:
            timings["graph_capture_s"] = time.monotonic() - start

    def timed_convert(*convert_args, **convert_kwargs):
        start = time.monotonic()
        try:
            return original_convert(*convert_args, **convert_kwargs)
        finally:
            timings["convert_s"] = time.monotonic() - start

    torch.jit.trace = timed_trace
    team.convert_to_coreml = timed_convert
    try:
        team.convert(
            str(checkpoint),
            ModelVersion.SD15,
            str(mlpackage_path),
            batch_size=args.batch_size,
            sample_size=(args.resolution // 8, args.resolution // 8),
            attn_impl=args.attention,
            quantize_nbits=quantize,
        )
    finally:
        torch.jit.trace = original_trace
        team.convert_to_coreml = original_convert

    if args.compile:
        compiled_path = mlpackage_path.with_suffix(".mlmodelc")
        compile_start = time.monotonic()
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(mlpackage_path), str(mlpackage_path.parent)],
            check=True,
        )
        generated_path = mlpackage_path.parent / f"{mlpackage_path.stem}.mlmodelc"
        if generated_path.exists() and generated_path != compiled_path:
            generated_path.rename(compiled_path)
        timings["first_load_compile_s"] = time.monotonic() - compile_start
        timings["mlmodelc_path"] = _relativize(compiled_path)

    timings["status"] = "ok"
    return timings


def _relativize(path) -> str:
    """Return ``path`` relative to the current workspace root so the emitted
    JSON sidecars stay portable (no $HOME leak when shared with a maintainer).
    Falls back to the absolute path when ``path`` is outside the cwd tree."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


def _quantize_arg(precision: str) -> str:
    if precision == "fp16":
        return "none"
    if precision.startswith("w"):
        return precision[1:]
    raise ValueError(f"Unsupported precision: {precision}")


if __name__ == "__main__":
    main()
