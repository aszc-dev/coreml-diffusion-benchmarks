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

app = typer.Typer(no_args_is_help=True)


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
    shared_input: Annotated[Path, typer.Option("--shared-input")] = Path("assets/shared_input/shared_input.npz"),
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
) -> None:
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
