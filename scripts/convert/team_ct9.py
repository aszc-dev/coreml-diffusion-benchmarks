import argparse
import json
import subprocess
import time
from pathlib import Path


# Per-component conversion settings. The UNet keeps its historical naming/behaviour
# (the existing artifacts and the convert orchestrator depend on it); the VAE decoder
# and text encoder are batch=1 and resolution rules differ (the text encoder is
# resolution-independent). See HANDOFF_VAE_CLIP_E2E.md A.2 and IMAGE_ABLATION_SPEC.md.
COMPONENTS = ("unet", "vae_decoder", "text_encoder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SD 1.5 components with the coreml-diffusion coremltools 9 toolchain.")
    parser.add_argument("--checkpoint", required=True, help="Local SD 1.5 .safetensors checkpoint.")
    parser.add_argument("--output-dir", default="artifacts/coreml_diffusion", help="Directory for CoreML artifacts.")
    parser.add_argument("--component", choices=[*COMPONENTS, "all"], default="unet",
                        help="Which component to convert. 'all' emits the UNet, VAE decoder and text encoder.")
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


def _component_spec(args: argparse.Namespace, component: str, output_dir: Path) -> dict:
    """Resolve the per-component conversion parameters, artifact path and quantize arg.

    The UNet carries the guided CFG batch and the attention variant in its name; the
    VAE decoder is batch=1 and resolution-dependent; the text encoder is batch=1 and
    resolution-independent (its package is shared across resolutions) and stays fp16.
    """
    if component == "unet":
        quantize = _quantize_arg(args.precision)
        return {
            "component": "unet",
            "mlpackage_path": output_dir / f"{args.model_label}-{args.attention}-{args.precision}.mlpackage",
            "batch_size": args.batch_size,
            "sample_size": (args.resolution // 8, args.resolution // 8),
            "quantize": quantize,
            "attn_impl": args.attention,
        }
    if component == "vae_decoder":
        quantize = _quantize_arg(args.precision)
        return {
            "component": "vae_decoder",
            "mlpackage_path": output_dir / f"{args.model_label}-vae_decoder-{args.resolution}-{args.precision}.mlpackage",
            "batch_size": 1,
            "sample_size": (args.resolution // 8, args.resolution // 8),
            "quantize": quantize,
            "attn_impl": None,
        }
    if component == "text_encoder":
        return {
            "component": "text_encoder",
            "mlpackage_path": output_dir / f"{args.model_label}-text_encoder.mlpackage",
            "batch_size": 1,
            "sample_size": (args.resolution // 8, args.resolution // 8),
            "quantize": "none",
            "attn_impl": None,
        }
    raise ValueError(f"Unknown component: {component}")


def _plan(args: argparse.Namespace, spec: dict, output_dir: Path, timings_out: Path) -> dict:
    return {
        "backend": "coreml_diffusion",
        "coremltools_major": 9,
        # `--dry-run` plan still shows the absolute paths since this is purely
        # a local debugging dump; the committed sidecar drops them.
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "output_dir": str(output_dir),
        "component": spec["component"],
        "mlpackage_path": str(spec["mlpackage_path"]),
        "attention": args.attention,
        "compute_unit": args.compute_unit,
        "resolution": args.resolution,
        "precision": args.precision,
        "quantize": spec["quantize"],
        "batch_size": spec["batch_size"],
        "compile": args.compile,
        "timings_out": str(timings_out),
    }


def _timings_path(args: argparse.Namespace, spec: dict, output_dir: Path) -> Path:
    if args.timings_out and args.component != "all":
        return Path(args.timings_out)
    if spec["component"] == "unet":
        # Preserve the historical UNet sidecar name (consumed by the orchestrator).
        return output_dir / f"{args.model_label}-{args.attention}-{args.compute_unit}-{args.precision}-conversion.json"
    return output_dir / f"{args.model_label}-{spec['component']}-{args.compute_unit}-conversion.json"


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output_dir)
    components = list(COMPONENTS) if args.component == "all" else [args.component]

    specs = [_component_spec(args, c, output_dir) for c in components]
    plans = [_plan(args, s, output_dir, _timings_path(args, s, output_dir)) for s in specs]
    if args.dry_run:
        # A single component prints one plan object (back-compat); `all` prints a list.
        print(json.dumps(plans[0] if len(plans) == 1 else plans, indent=2, sort_keys=True))
        return

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        timings_out = _timings_path(args, spec, output_dir)
        timings = convert_component(args, checkpoint, spec)
        timings_out.parent.mkdir(parents=True, exist_ok=True)
        timings_out.write_text(json.dumps(timings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote conversion timings to {timings_out}")


def convert_component(args: argparse.Namespace, checkpoint: Path, spec: dict) -> dict:
    import torch
    import coreml_diffusion.convert as team
    from coreml_diffusion.model_version import ModelVersion

    component = spec["component"]
    mlpackage_path = spec["mlpackage_path"]
    timings: dict[str, float | str | None] = {
        "backend": "coreml_diffusion",
        "coremltools_major": 9,
        # NOTE: full checkpoint path intentionally omitted from this sidecar
        # ($HOME leak). The checkpoint SHA-256 lives in .source.json next to
        # the artifact and is the canonical identity for reproducibility.
        "component": component,
        "attention": args.attention,
        "compute_unit": args.compute_unit,
        "resolution": args.resolution,
        "precision": args.precision,
        "quantize": spec["quantize"],
        "batch_size": spec["batch_size"],
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

    convert_kwargs = {
        "batch_size": spec["batch_size"],
        "sample_size": spec["sample_size"],
        "quantize_nbits": spec["quantize"],
    }
    if component == "unet":
        # Keep the historical UNet call byte-identical (component defaults to "unet").
        convert_kwargs["attn_impl"] = spec["attn_impl"]
    else:
        convert_kwargs["component"] = component

    torch.jit.trace = timed_trace
    team.convert_to_coreml = timed_convert
    try:
        team.convert(str(checkpoint), ModelVersion.SD15, str(mlpackage_path), **convert_kwargs)
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
