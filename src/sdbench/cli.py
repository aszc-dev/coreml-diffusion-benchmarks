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
from sdbench.inputs import generate_shared_input, load_shared_input, save_shared_input
from sdbench.orchestrator import run_matrix
from sdbench.power import apply_power_to_records, parse_powermetrics_plist
from sdbench.results import load_jsonl, write_jsonl, write_summary_tables

app = typer.Typer()


_SUPPORTED_PYTHON = (3, 12)


def _check_python_version() -> None:
    """Refuse to start on a Python the heavy stack can't bind to.

    coremltools 8.3.0 ships native wheels only for Python 3.10/3.11/3.12;
    on 3.13+ ``coremltools.libcoremlpython._MLModelProxy`` ends up None and
    the first adapter load fails with the cryptic
    "Unable to load any compiled models. This is most likely because
    coremltools was installed from an egg rather than a wheel." We catch
    that here with an actionable remedy instead.
    """
    import sys

    actual = sys.version_info[:2]
    if actual != _SUPPORTED_PYTHON:
        typer.echo(
            f"[cdbench] Python {actual[0]}.{actual[1]} is not supported "
            f"(coremltools 8.3 has no wheel for it). Use Python 3.12:\n"
            f"  uv tool uninstall coreml-diffusion-benchmarks && \\\n"
            f"  uv tool install --python 3.12 coreml-diffusion-benchmarks\n"
            f"or, from a clone:\n"
            f"  uv sync --python 3.12",
            err=True,
        )
        raise typer.Exit(code=2)


@app.callback(invoke_without_command=True)
def _entry(ctx: typer.Context) -> None:
    """SD 1.5 UNet cross-backend benchmark. Run with no command for the guided flow."""
    _check_python_version()
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
    force_power: Annotated[bool, typer.Option("--force-power", help="Record power even if the env check fails (battery/low-power/noisy host). The numbers will be flagged.")] = False,
    repeats: Annotated[int | None, typer.Option("--repeats", help="Repeat the matrix N times as one session; per-cell median + p10/p90 of latency, power and energy are aggregated to sessions/<id>/aggregated.jsonl. Defaults to the saved run plan; pass 1 to force a single pass. Use N>=3 to characterise between-run noise on energy (R5.4).")] = None,
    cooldown_s: Annotated[float | None, typer.Option("--cooldown-s", help="Seconds to sleep between repeats (then gate on thermal state). Ignored when repeats=1. Defaults to the saved run plan.")] = None,
) -> None:
    """Run the benchmark with live progress, minimal-root power, and upserted results."""
    from sdbench.tui.run_cmd import run_benchmark, run_session
    from sdbench.tui.runplan import load_runplan
    from sdbench.tui.workspace import Workspace

    ws = Workspace.resolve(workspace)
    # Fall back to the saved plan's multi-run settings when the CLI didn't
    # override them — the TUI's default plan is publication-grade
    # (repeats=7), and a bare ``sdbench run`` ought to honour that rather
    # than silently regress to single-pass.
    if use_plan and ws.runplan_path.exists():
        try:
            plan = load_runplan(ws.runplan_path)
            if repeats is None:
                repeats = plan.repeats
            if cooldown_s is None:
                cooldown_s = plan.cooldown_s
        except (OSError, ValueError, KeyError):
            pass  # malformed plan — defaults below apply.
    if repeats is None:
        repeats = 1
    if cooldown_s is None:
        cooldown_s = 30.0
    if repeats > 1:
        run_session(
            ws,
            config,
            repeats=repeats,
            cooldown_s=cooldown_s,
            cell_ids=cell or None,
            power=power,
            verbosity=verbosity,
            use_plan=use_plan,
            force_power=force_power,
        )
        return
    run_benchmark(
        ws,
        config,
        cell_ids=cell or None,
        power=power,
        verbosity=verbosity,
        use_plan=use_plan,
        force_power=force_power,
    )


@app.command("run-matrix")
def run_matrix_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config/matrix.yaml"),
    shared_input: Annotated[Path, typer.Option("--shared-input")] = Path("assets/shared_input/shared_input.npz"),
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
) -> None:
    """Headless engine run (no harness-side power, upsert-write). Used by scripts/run.sh and CI.

    Delegates to :func:`sdbench.tui.run_cmd.run_benchmark` so every record gets the
    full reproducibility telemetry (host_id_hash, provenance_digest, latent SHA,
    env_vars_digest, conversion timings). Power is disabled on the harness side
    because ``scripts/run.sh`` runs its own external ``powermetrics`` sampler;
    ``sdbench power`` post-processes the resulting plist."""
    from sdbench.tui.run_cmd import run_benchmark
    from sdbench.tui.workspace import Workspace

    ws = Workspace.resolve(results_dir.parent if results_dir.name == "results" else None)
    run_id = os.environ.get("SDBENCH_RUN_ID") or str(uuid4())
    shared = load_shared_input(shared_input)
    records = run_benchmark(
        ws,
        config,
        cell_ids=None,         # falls back to cfg.enabled_cells()
        power=False,           # external sampler — see scripts/run.sh
        verbosity="quiet",
        use_plan=False,
        run_id=run_id,
        shared_input=shared,
    )
    typer.echo(f"Wrote {len(records)} records to {ws.results_data_dir / 'results.jsonl'}")


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
    manifest = _load_manifest_for_tables(input_path.parent / "environment.json")
    write_summary_tables(records, output_dir, manifest=manifest)
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


@app.command("report")
def report(
    run_id: Annotated[str | None, typer.Option("--run-id", help="Bundle this run id (defaults to manifest.run_id).")] = None,
    output_root: Annotated[Path | None, typer.Option("--output-root", help="Where the bundle lands (default: results/reports).")] = None,
    zip_bundle: Annotated[bool, typer.Option("--zip/--no-zip", help="Also write a .zip alongside the bundle directory.")] = True,
    anonymize: Annotated[bool, typer.Option("--anonymize/--no-anonymize", help="Strip filesystem-local PII and re-hash the host id.")] = False,
    salt: Annotated[str | None, typer.Option("--salt", help="Required when --anonymize is set.")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace")] = None,
) -> None:
    """Bundle a run for contributor submission (manifest + JSONL + plist + tables + matrix)."""
    from sdbench.report import build_report
    from sdbench.tui.workspace import Workspace

    ws = Workspace.resolve(workspace)
    path = build_report(
        ws,
        run_id=run_id,
        output_root=output_root,
        zip_bundle=zip_bundle,
        anonymize=anonymize,
        salt=salt,
    )
    typer.echo(f"Wrote contributor bundle to {path}")


@app.command("validate-report")
def validate_report_command(
    bundle: Annotated[Path, typer.Argument(help="Path to a bundle directory (not the .zip).")],
) -> None:
    """Validate a contributor bundle: schema version + digest + SHA consistency (R11.14, A11)."""
    from sdbench.report import validate_report

    result = validate_report(bundle)
    spread = f"{result.energy_spread_max:.3f}" if result.energy_spread_max is not None else "N/A"
    n_ok_min = result.n_runs_ok_min if result.n_runs_ok_min is not None else "N/A"
    overrides = ",".join(result.matrix_overrides) or "none"
    typer.echo(
        f"schema_version={result.schema_version} supported={result.supported_schema} "
        f"schema_ok={result.schema_ok} digests_consistent={result.digests_consistent} "
        f"digests_match_manifest={result.digests_match_manifest} "
        f"latent_consistent={result.latent_consistent} text_embedding_consistent={result.text_embedding_consistent} "
        f"session_id={result.session_id or 'N/A'} n_runs_ok_min={n_ok_min} "
        f"energy_spread_max={spread} matrix_overrides={overrides} "
        f"session_ok={result.session_ok} ok={result.ok}"
    )
    for issue in result.issues:
        typer.echo(f"  - {issue}")
    raise typer.Exit(0 if result.ok else 1)


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
    power_log: Annotated[Path, typer.Option("--power-log", help="Single-pass plist. Ignored when --session is given (each pass has its own plist).")] = Path("results/raw/none.plist"),
    input_path: Annotated[Path, typer.Option("--input", "-i")] = Path("results/data/results.jsonl"),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config/matrix.yaml"),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("results/tables"),
    session: Annotated[Path | None, typer.Option("--session", help="Re-apply power to a whole multi-run session directory (sessions/<id>/). Each pass is realigned from its own plist in --raw-dir, then re-aggregated. This is what the report validator reads, so single-file --input mode does NOT fix a multi-run bundle.")] = None,
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="Directory holding per-run <run_id>-powermetrics.plist files (for --session).")] = Path("results/raw"),
) -> None:
    """Align a powermetrics capture to the timed windows and fill power into the records.

    Run after the benchmark (the sampler runs concurrently, so power is post-hoc). (R6.2-R6.4)

    Two modes:
    - default: realign one --power-log plist into a single --input JSONL.
    - --session: realign every pass JSONL in a session dir from its own plist
      and rebuild sessions/<id>/aggregated.jsonl, which is the artifact the
      report validator checks for energy spread (R5.4)."""
    cfg = load_benchmark_config(config)
    if session is not None:
        _apply_power_to_session(session, raw_dir, cfg, output_dir)
        return
    records = load_jsonl(input_path)
    samples = parse_powermetrics_plist(power_log)
    updated = apply_power_to_records(records, samples, cfg.power.baseline_seconds, cfg.iterations)
    write_jsonl(updated, input_path)
    # Re-regenerated tables must keep their captions; load the manifest the
    # preceding `sdbench run` / `run-matrix` wrote so the chip / build /
    # provenance digest land in every table header.
    manifest_path = input_path.parent / "environment.json"
    manifest = _load_manifest_for_tables(manifest_path)
    write_summary_tables(updated, output_dir, manifest=manifest)
    typer.echo(f"Applied {len(samples)} power samples to {len(updated)} records in {input_path}")


def _apply_power_to_session(session_dir: Path, raw_dir: Path, cfg: BenchmarkConfig, output_dir: Path) -> None:
    """Realign every pass in a session from its own plist, then re-aggregate.

    The multi-run aggregate (sessions/<id>/aggregated.jsonl) is computed from the
    per-pass JSONLs, so realigning only the top-level results.jsonl leaves the
    aggregate — and the validator's energy-spread check — on stale numbers. We
    realign each run-NN.jsonl from results/raw/<run_id>-powermetrics.plist, then
    recompute the aggregate the same way run_session does."""
    from sdbench.aggregate import aggregate_session

    pass_paths = sorted(session_dir.glob("run-*.jsonl"))
    if not pass_paths:
        raise typer.BadParameter(f"no run-*.jsonl passes under {session_dir}")
    pass_records: list[list] = []
    n_samples = 0
    for pass_path in pass_paths:
        records = load_jsonl(pass_path)
        run_id = next((r.run_id for r in records if r.run_id), None)
        plist = raw_dir / f"{run_id}-powermetrics.plist"
        if not plist.exists():
            typer.echo(f"[warn] {pass_path.name}: plist {plist.name} missing — leaving pass unchanged")
            pass_records.append(records)
            continue
        samples = parse_powermetrics_plist(plist)
        n_samples += len(samples)
        updated = apply_power_to_records(records, samples, cfg.power.baseline_seconds, cfg.iterations)
        write_jsonl(updated, pass_path)
        pass_records.append(updated)
    flat = [rec for sub in pass_records for rec in sub]
    aggregated = aggregate_session(flat)
    write_jsonl(aggregated, session_dir / "aggregated.jsonl")
    # Tables are the human-facing view; regenerate them from the *aggregate*
    # (one median row per cell) so they don't keep showing the stale per-pass
    # dump. Resolve the environment manifest next to the bundle root (two levels
    # up from sessions/<id>/) so captions survive.
    manifest = _load_manifest_for_tables(session_dir.parent.parent / "tables" / "environment.json")
    if manifest is None:
        manifest = _load_manifest_for_tables(session_dir.parent.parent / "environment.json")
    write_summary_tables(aggregated, output_dir, manifest=manifest)
    typer.echo(
        f"Realigned {len(pass_paths)} passes ({n_samples} samples), rebuilt "
        f"{session_dir / 'aggregated.jsonl'} and tables in {output_dir}"
    )


def _load_manifest_for_tables(path: Path):
    """Best-effort load of the environment manifest as a SimpleNamespace tree.

    Used so post-hoc table writers (``sdbench power``, ``sdbench tables``) can
    surface the table captions and the environment.md appendix without
    re-running the benchmark. Returns None if the manifest is missing or
    unreadable — captions are simply omitted in that case.
    """
    if not path.exists():
        return None
    try:
        import json
        from types import SimpleNamespace

        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    def _to_ns(value):
        if isinstance(value, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in value.items()})
        if isinstance(value, list):
            return [_to_ns(v) for v in value]
        return value

    return _to_ns(raw)


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
