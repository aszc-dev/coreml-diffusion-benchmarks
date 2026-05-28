import os

# Silence chatty libraries at the SOURCE before anything imports them.
# Even with fd/Python output capture in place, some libs (huggingface_hub
# progress bars, tqdm, coremltools logging) write through handles or paths that
# bypass our redirects and corrupt the full-screen Live. Off > captured.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer

from sdbench.backends.registry import build_default_adapters
from sdbench.config import BenchmarkConfig, load_benchmark_config
from sdbench.env import collect_environment_manifest, write_environment_manifest
from sdbench.inputs import generate_shared_input, load_shared_input, save_shared_input
from sdbench.orchestrator import run_matrix
from sdbench.power import apply_power_to_records, parse_powermetrics_plist
from sdbench.results import load_jsonl, write_jsonl, write_summary_tables

app = typer.Typer()


@app.callback(invoke_without_command=True)
def _entry(ctx: typer.Context) -> None:
    """SD 1.5 UNet cross-backend benchmark. Run with no command for the guided flow."""
    if ctx.invoked_subcommand is not None:
        return
    from sdbench.tui.app import guided_main
    from sdbench.tui.workspace import Workspace

    guided_main(Workspace.resolve(None), Path("config/matrix.yaml"))


@app.command("prepare-input")
def prepare_input(
    config: Annotated[Path, typer.Option("--config", "-c")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("assets/shared_input/shared_input.npz"),
) -> None:
    cfg = load_benchmark_config(config)
    resolution = _single_resolution(cfg)
    shared = generate_shared_input(seed=cfg.seed, resolution=resolution)
    save_shared_input(shared, output)
    typer.echo(f"Wrote shared input to {output}")


@app.command("run")
def run(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config/matrix.yaml"),
    workspace: Annotated[Path | None, typer.Option("--workspace", help="Workspace root (defaults to the current directory).")] = None,
    cell: Annotated[list[str] | None, typer.Option("--cell", help="Run only these cell ids (repeatable). Overrides the saved run plan.")] = None,
    power: Annotated[bool | None, typer.Option("--power/--no-power", help="Measure power (only the sampler runs as root). Defaults to the run plan.")] = None,
    verbosity: Annotated[str | None, typer.Option("--verbosity", help="quiet | normal | verbose.")] = None,
    use_plan: Annotated[bool, typer.Option("--use-plan/--no-use-plan", help="Use the run plan saved by `config`.")] = True,
) -> None:
    """Run the benchmark with live progress, minimal-root power, and upserted results."""
    from sdbench.tui.run_cmd import run_benchmark
    from sdbench.tui.workspace import Workspace

    run_benchmark(
        Workspace.resolve(workspace),
        config,
        cell_ids=cell or None,
        power=power,
        verbosity=verbosity,
        use_plan=use_plan,
    )


@app.command("run-matrix")
def run_matrix_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config/matrix.yaml"),
    shared_input: Annotated[Path, typer.Option("--shared-input")] = Path("assets/shared_input/shared_input.npz"),
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
) -> None:
    """Headless engine run (no power, clobber-write). Used by scripts/run.sh and CI."""
    cfg = load_benchmark_config(config)
    records = run_matrix(
        cfg=_with_cells(cfg, cfg.enabled_cells()),
        shared_input=load_shared_input(shared_input),
        adapters=build_default_adapters(checkpoint_path=cfg.checkpoint),
        run_id=str(uuid4()),
        results_dir=results_dir,
    )
    data_path = results_dir / "data" / "results.jsonl"
    write_jsonl(records, data_path)
    write_summary_tables(records, results_dir / "tables")
    manifest = collect_environment_manifest(
        seed=cfg.seed,
        run_conditions="default CLI run; record background workload manually for publication runs",
        checkpoint_path=cfg.checkpoint,
    )
    write_environment_manifest(manifest, results_dir / "data" / "environment.json")
    typer.echo(f"Wrote {len(records)} records to {data_path}")


@app.command("run-cell")
def run_cell(
    cell_id: Annotated[str | None, typer.Option("--cell")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    compute_unit: Annotated[str | None, typer.Option("--compute-unit")] = None,
    attention: Annotated[str, typer.Option("--attention")] = "NATIVE",
    precision: Annotated[str, typer.Option("--precision")] = "fp16",
    resolution: Annotated[int, typer.Option("--resolution")] = 512,
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config/matrix.yaml"),
    shared_input: Annotated[Path, typer.Option("--shared-input")] = Path("assets/shared_input/shared_input.npz"),
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
) -> None:
    cfg = load_benchmark_config(config)
    if cell_id is not None:
        cell = cfg.select_cell_by_id(cell_id)
    else:
        if backend is None or compute_unit is None:
            raise typer.BadParameter("Provide either --cell or both --backend and --compute-unit")
        cell = cfg.select_cell(backend, compute_unit, attention, precision, resolution)
    selected_cfg = _with_cells(cfg, [cell])
    records = run_matrix(
        cfg=selected_cfg,
        shared_input=load_shared_input(shared_input),
        adapters=build_default_adapters(checkpoint_path=cfg.checkpoint),
        run_id=str(uuid4()),
        results_dir=results_dir,
    )
    data_path = results_dir / "data" / f"{cell.id}.jsonl"
    write_jsonl(records, data_path)
    write_summary_tables(records, results_dir / "tables")
    typer.echo(f"Wrote cell record to {data_path}")


@app.command("tables")
def tables(
    input_path: Annotated[Path, typer.Option("--input", "-i")] = Path("results/data/results.jsonl"),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("results/tables"),
) -> None:
    records = load_jsonl(input_path)
    write_summary_tables(records, output_dir)
    typer.echo(f"Wrote summary tables to {output_dir}")


@app.command("download")
def download(
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", help="Local SD 1.5 .safetensors to verify and use.")] = None,
    auto: Annotated[bool, typer.Option("--download/--no-download", help="Fetch from the official HF repo if not present.")] = False,
    workspace: Annotated[Path | None, typer.Option("--workspace")] = None,
) -> None:
    """Resolve and SHA-verify the SD 1.5 checkpoint (point at a local file or auto-download)."""
    from sdbench.tui.download import resolve_checkpoint
    from sdbench.tui.workspace import Workspace

    path = resolve_checkpoint(Workspace.resolve(workspace), explicit=checkpoint, auto_download=auto)
    typer.echo(f"Verified checkpoint: {path}")


@app.command("convert")
def convert(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config/matrix.yaml"),
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", help="Override the checkpoint to convert from.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if a cached artifact matches the checkpoint.")] = False,
    workspace: Annotated[Path | None, typer.Option("--workspace")] = None,
) -> None:
    """Convert CoreML artifacts in the isolated ct8/ct9 envs (cached by checkpoint SHA)."""
    from sdbench.provenance import sha256_file
    from sdbench.tui.convert_orchestrator import convert_all
    from sdbench.tui.workspace import Workspace

    ws = Workspace.resolve(workspace)
    cfg = load_benchmark_config(config)
    ckpt = checkpoint or cfg.checkpoint
    ckpt_sha = sha256_file(ckpt) if Path(ckpt).is_file() else None
    ran = convert_all(ws, cfg, ckpt, ckpt_sha, force=force)
    typer.echo(f"Converted {len(ran)} build(s); the rest were cached.")


@app.command("measure-disk")
def measure_disk(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config/matrix.yaml"),
    workspace: Annotated[Path | None, typer.Option("--workspace")] = None,
) -> None:
    """Measure converted-artifact sizes and write config/disk_footprint.yaml (no guesswork)."""
    from sdbench.tui.sizing_probe import measure_cell_footprint, write_footprint
    from sdbench.tui.workspace import Workspace

    ws = Workspace.resolve(workspace)
    cfg = load_benchmark_config(config)
    sizes = measure_cell_footprint(ws, cfg)
    write_footprint(ws.disk_footprint_path, sizes)
    typer.echo(f"Measured {len(sizes)} artifact(s) → {ws.disk_footprint_path}")


@app.command("verify")
def verify(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config/matrix.yaml"),
    workspace: Annotated[Path | None, typer.Option("--workspace")] = None,
) -> None:
    """Check that all results share one provenance fingerprint matching this environment."""
    from sdbench.provenance import collect_fingerprint, sha256_file, verify_results
    from sdbench.results import load_jsonl
    from sdbench.tui.workspace import Workspace

    ws = Workspace.resolve(workspace)
    cfg = load_benchmark_config(config)
    results = ws.results_data_dir / "results.jsonl"
    if not results.exists():
        typer.echo("No results to verify.")
        raise typer.Exit(1)
    sha = sha256_file(cfg.checkpoint) if Path(cfg.checkpoint).is_file() else None
    report = verify_results(load_jsonl(results), collect_fingerprint(ws, sha))
    typer.echo(f"records={report.total} distinct_digests={len(report.digests)} ok={report.ok}")
    raise typer.Exit(0 if report.ok else 1)


@app.command("config")
def config_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config/matrix.yaml"),
    workspace: Annotated[Path | None, typer.Option("--workspace", help="Workspace root (defaults to the current directory).")] = None,
) -> None:
    """Interactively choose cells, power, and verbosity; save a run plan for `run`."""
    from sdbench.tui.config_view import run_config_screen
    from sdbench.tui.workspace import Workspace

    run_config_screen(Workspace.resolve(workspace), config)


@app.command("cleanup")
def cleanup(
    select_all: Annotated[bool, typer.Option("--all", help="Select every reclaimable target.")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Reclaim only these target keys (repeatable): artifacts, power-raw, results, shared-input, cache."),
    ] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", help="Workspace root (defaults to the current directory).")] = None,
) -> None:
    """Reclaim generated benchmark state (converted models, results, captures) with measured sizes."""
    from sdbench.tui.cleanup import run_cleanup
    from sdbench.tui.workspace import Workspace

    run_cleanup(Workspace.resolve(workspace), select_all=select_all, assume_yes=yes, only=only)


@app.command("power")
def power(
    power_log: Annotated[Path, typer.Option("--power-log")],
    input_path: Annotated[Path, typer.Option("--input", "-i")] = Path("results/data/results.jsonl"),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config/matrix.yaml"),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("results/tables"),
) -> None:
    """Align a powermetrics capture to the timed windows and fill power into the records.

    Run after the benchmark (the sampler runs concurrently, so power is post-hoc). (R6.2-R6.4)"""
    cfg = load_benchmark_config(config)
    records = load_jsonl(input_path)
    samples = parse_powermetrics_plist(power_log)
    updated = apply_power_to_records(records, samples, cfg.power.baseline_seconds, cfg.iterations)
    write_jsonl(updated, input_path)
    write_summary_tables(updated, output_dir)
    typer.echo(f"Applied {len(samples)} power samples to {len(updated)} records in {input_path}")


def _single_resolution(cfg: BenchmarkConfig) -> int:
    resolutions = {cell.resolution for cell in cfg.cells}
    if len(resolutions) != 1:
        raise ValueError("Shared input generation requires one resolution per invocation")
    return resolutions.pop()


def _with_cells(cfg: BenchmarkConfig, cells) -> BenchmarkConfig:
    return BenchmarkConfig(
        checkpoint=cfg.checkpoint,
        seed=cfg.seed,
        iterations=cfg.iterations,
        warmup=cfg.warmup,
        thermal=cfg.thermal,
        equivalence=cfg.equivalence,
        power=cfg.power,
        cells=list(cells),
    )


if __name__ == "__main__":
    app()
