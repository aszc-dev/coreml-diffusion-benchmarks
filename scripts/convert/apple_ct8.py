import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SD 1.5 UNet with Apple's coremltools 8 toolchain.")
    parser.add_argument("--checkpoint", required=True, help="Local SD 1.5 .safetensors checkpoint.")
    parser.add_argument("--output-dir", default="artifacts/apple_coreml", help="Directory for CoreML artifacts.")
    parser.add_argument("--attention", choices=["ORIGINAL", "SPLIT_EINSUM", "SPLIT_EINSUM_V2"], required=True)
    parser.add_argument("--compute-unit", choices=["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL"], required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model-label", default="local_sd15")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timings-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output_dir)
    timings_out = Path(args.timings_out) if args.timings_out else output_dir / f"{args.model_label}-{args.attention}-{args.compute_unit}-conversion.json"

    plan = {
        "backend": "apple_coreml",
        "coremltools_major": 8,
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "attention": args.attention,
        "compute_unit": args.compute_unit,
        "resolution": args.resolution,
        "compile": args.compile,
        "timings_out": str(timings_out),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)

    timings = convert_unet(args, checkpoint, output_dir)
    timings_out.parent.mkdir(parents=True, exist_ok=True)
    timings_out.write_text(json.dumps(timings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote conversion timings to {timings_out}")


def convert_unet(args: argparse.Namespace, checkpoint: Path, output_dir: Path) -> dict:
    import torch
    from diffusers import StableDiffusionPipeline
    import python_coreml_stable_diffusion.torch2coreml as apple
    from python_coreml_stable_diffusion import unet as apple_unet

    tool_args = apple.parser_spec().parse_args(
        [
            "--model-version",
            args.model_label,
            "--convert-unet",
            "--attention-implementation",
            args.attention,
            "--compute-unit",
            args.compute_unit,
            "--latent-h",
            str(args.resolution // 8),
            "--latent-w",
            str(args.resolution // 8),
            "-o",
            str(output_dir),
        ]
    )
    apple_unet.ATTENTION_IMPLEMENTATION_IN_EFFECT = apple_unet.AttentionImplementations[args.attention]

    timings: dict[str, float | str | None] = {
        "backend": "apple_coreml",
        "coremltools_major": 8,
        "checkpoint": str(checkpoint),
        "attention": args.attention,
        "compute_unit": args.compute_unit,
        "resolution": args.resolution,
        "graph_capture_s": None,
        "convert_s": None,
        "first_load_compile_s": None,
        "mlpackage_path": None,
        "mlmodelc_path": None,
        "status": "started",
    }

    pipe_start = time.monotonic()
    pipe = StableDiffusionPipeline.from_single_file(
        str(checkpoint),
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    timings["pipeline_load_s"] = time.monotonic() - pipe_start

    original_trace = torch.jit.trace
    original_convert = apple._convert_to_coreml

    def timed_trace(*trace_args, **trace_kwargs):
        start = time.monotonic()
        try:
            return original_trace(*trace_args, **trace_kwargs)
        finally:
            timings["graph_capture_s"] = time.monotonic() - start

    def timed_convert(*convert_args, **convert_kwargs):
        start = time.monotonic()
        try:
            model, out_path = original_convert(*convert_args, **convert_kwargs)
            timings["mlpackage_path"] = out_path
            return model, out_path
        finally:
            timings["convert_s"] = time.monotonic() - start

    torch.jit.trace = timed_trace
    apple._convert_to_coreml = timed_convert
    try:
        apple.convert_unet(pipe, tool_args)
    finally:
        torch.jit.trace = original_trace
        apple._convert_to_coreml = original_convert

    if args.compile:
        mlpackage_path = Path(str(timings["mlpackage_path"]))
        compiled_path = output_dir / f"{mlpackage_path.stem}.mlmodelc"
        compile_start = time.monotonic()
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(mlpackage_path), str(output_dir)],
            check=True,
        )
        generated_path = output_dir / f"{mlpackage_path.stem}.mlmodelc"
        if generated_path.exists() and generated_path != compiled_path:
            generated_path.rename(compiled_path)
        timings["first_load_compile_s"] = time.monotonic() - compile_start
        timings["mlmodelc_path"] = str(compiled_path)

    timings["status"] = "ok"
    return timings


if __name__ == "__main__":
    main()
