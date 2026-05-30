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
    # Provenance fingerprint digest of the run that produced this record (R10.4/R11.3).
    # Lets a datapoint be traced to its checkpoint + pinned dependency set.
    provenance_digest: str | None = None
    # Reproducibility telemetry stamped per-cell so a single row is self-describing
    # about the host and the runtime conditions it ran under (R11.6-R11.13).
    # All defaulted: older JSONL records still load.
    schema_version: int | None = None
    host_id_hash: str | None = None
    thermal_at_cell_start: dict | None = None
    thermal_at_cell_end: dict | None = None
    loadavg_at_cell_start: list[float] | None = None
    env_vars_digest: str | None = None
    power_sampler_interval_ms: int | None = None
    latent_input_sha256: str | None = None
    text_embedding_input_sha256: str | None = None
    # Multi-run session telemetry (schema v3, R5.4 between-run variance).
    # ``session_id`` spans the N independent runs of ``sdbench run --repeats N``;
    # ``run_id`` still identifies a single matrix pass within that session.
    # ``repeat_index`` is 0..repeat_count-1 within the session; both default to
    # None so single-run records and pre-v3 JSONL keep loading unchanged.
    session_id: str | None = None
    repeat_index: int | None = None
    repeat_count: int | None = None
    # Aggregated-row fields. Populated only by ``sdbench.aggregate`` for the
    # per-cell rows in ``sessions/<id>/aggregated.jsonl``; the median columns
    # above keep their meaning ("median across the N passes") while p10/p90 add
    # the between-run spread. ``n_runs_ok``/``failed`` describe how many passes
    # this row was reduced from.
    latency_ms_p10: float | None = None
    latency_ms_p90: float | None = None
    gpu_power_w_p10: float | None = None
    gpu_power_w_p90: float | None = None
    ane_power_w_p10: float | None = None
    ane_power_w_p90: float | None = None
    energy_per_unet_step_j_p10: float | None = None
    energy_per_unet_step_j_p90: float | None = None
    n_runs_ok: int | None = None
    n_runs_failed: int | None = None


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


def upsert_jsonl(records: list[BenchmarkRecord], path: str | Path, key=None) -> list[BenchmarkRecord]:
    """Merge records into an existing JSONL, replacing rows with a matching key.

    Keyed by ``(cell_id, repeat_index)`` by default: running a single cell
    updates only its row and leaves the rest of the matrix intact, instead of
    clobbering the whole file. Existing rows keep their order; genuinely new
    cells are appended. ``repeat_index`` is treated as 0 when absent so single-
    run records and pre-v3 JSONL load unchanged — multi-run sessions
    (``--repeats N``) get one row per (cell_id, repeat_index) pair without
    overwriting earlier passes.
    """
    keyfn = key or (lambda record: (record.cell_id, record.repeat_index or 0))
    output = Path(path)
    merged: dict = {}
    if output.exists():
        for record in load_jsonl(output):
            merged[keyfn(record)] = record
    for record in records:
        merged[keyfn(record)] = record
    materialized = list(merged.values())
    write_jsonl(materialized, output)
    return materialized


def write_summary_tables(
    records: list[BenchmarkRecord],
    output_dir: str | Path,
    *,
    manifest=None,
) -> None:
    """Emit publication-ready tables. When ``manifest`` is given, prepend each
    table with a caption (tool/chip/macOS build/provenance digest) so a table
    copy-pasted into a blog post still carries its host context (R10.2, R11.6)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    caption = _table_caption(manifest)
    # When the records come from a session-aggregator pass, the median columns
    # mean "median across N repeats" and the spread columns expose the
    # between-run p10–p90 the reader needs to gauge reproducibility (R5.4).
    # Render the spread inline as ``median [p10–p90]`` so the table stays one
    # row per cell instead of doubling for the repeat dimension; if no
    # aggregation is present, fall back to the single-run "median (IQR)" shape.
    multi_run = any(record.repeat_count and record.repeat_index is None for record in records)
    if multi_run:
        n_runs_caption = _multi_run_caption(records)
        caption = f"{caption} · {n_runs_caption}" if caption else n_runs_caption
    _write_table(
        out / "latency.md",
        ["Cell", "Backend", "Median latency (ms)", "Within-pass IQR (ms)", "Between-pass p10–p90 (ms)", "n_ok", "Status"]
        if multi_run
        else ["Cell", "Backend", "Median latency (ms)", "IQR (ms)", "Status"],
        [
            (
                [
                    record.cell_id,
                    record.backend,
                    _fmt(record.latency_ms_median),
                    _fmt(record.latency_ms_iqr),
                    _fmt_range(record.latency_ms_p10, record.latency_ms_p90),
                    _fmt(record.n_runs_ok),
                    record.status,
                ]
                if multi_run
                else [
                    record.cell_id,
                    record.backend,
                    _fmt(record.latency_ms_median),
                    _fmt(record.latency_ms_iqr),
                    record.status,
                ]
            )
            for record in records
        ],
        caption=caption,
    )
    _write_table(
        out / "power_energy.md",
        [
            "Cell",
            "GPU power (W)",
            "ANE power (W)",
            "Energy / UNet step (J)" + (" · p10–p90" if multi_run else ""),
            "Estimated energy / 50-step image (J)",
        ],
        [
            [
                record.cell_id,
                _fmt_median_with_spread(record.gpu_power_w, record.gpu_power_w_p10, record.gpu_power_w_p90),
                _fmt_median_with_spread(record.ane_power_w, record.ane_power_w_p10, record.ane_power_w_p90),
                _fmt_median_with_spread(
                    record.energy_per_unet_step_j,
                    record.energy_per_unet_step_j_p10,
                    record.energy_per_unet_step_j_p90,
                ),
                _fmt(record.estimated_energy_per_50_step_image_j),
            ]
            for record in records
        ],
        caption=caption,
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
        caption=caption,
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
        caption=caption,
    )
    # Equivalence is its own table so the reader can audit numerical fidelity
    # without trawling JSONL. Float displays use scientific notation; rounding
    # to four decimals would render 0.9999996 as "1.0000" and hide real drift.
    _write_table(
        out / "equivalence.md",
        ["Cell", "MSE", "Cosine", "Flagged"],
        [
            [
                record.cell_id,
                _fmt_eq(record.mse, ".3e"),
                _fmt_eq(record.cosine, ".7f"),
                _fmt_flag(record.numerically_divergent),
            ]
            for record in records
        ],
        caption=caption,
    )
    if manifest is not None:
        write_environment_appendix(manifest, out / "environment.md")


def _field(obj, name, default=None):
    """Read ``name`` from either a dataclass / SimpleNamespace or a plain dict.

    Used so post-hoc table writers (``sdbench power``, ``sdbench tables``) can
    accept a manifest loaded from JSON (dict tree) AND one passed in-process
    (dataclass tree) without two code paths."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_mapping(obj) -> dict:
    """Coerce a dict-like-or-namespace child into a real dict for ``.items()``."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def _table_caption(manifest) -> str | None:
    if manifest is None:
        return None
    hw = _field(manifest, "hardware")
    chip = _field(hw, "chip_brand")
    os_info = _field(manifest, "os")
    os_label = None
    if os_info is not None:
        product = _field(os_info, "product_name", "") or ""
        version = _field(os_info, "product_version", "") or ""
        build = _field(os_info, "build_version", "") or ""
        os_label = f"{product} {version} ({build})".strip()
    tool = _field(manifest, "tool_version") or "sdbench"
    digest = _field(manifest, "provenance_digest")
    digest_label = f"provenance {digest[:12]}" if digest else None
    parts = [f"sdbench {tool}", chip, os_label, digest_label]
    return " · ".join(p for p in parts if p)


def _write_table(
    path: Path,
    headers: list[str],
    rows: list[list[str]],
    *,
    caption: str | None = None,
) -> None:
    lines: list[str] = []
    if caption:
        lines.append(f"<!-- {caption} -->")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_environment_appendix(manifest, path: str | Path) -> None:
    """Render the full manifest as a human-readable Markdown appendix (R11.14).

    Accepts both an in-process dataclass tree and a JSON-loaded dict tree, so the
    same writer is used by ``sdbench run`` and by post-hoc paths like
    ``sdbench tables`` / ``sdbench power`` that load ``environment.json``.
    """
    sections: list[str] = []
    sections.append("# Run environment")
    sections.append("")
    sections.append(f"- **schema_version**: `{_field(manifest, 'schema_version')}`")
    sections.append(f"- **tool_version**: `{_field(manifest, 'tool_version') or '(unknown)'}`")
    sections.append(f"- **run_id**: `{_field(manifest, 'run_id') or '(unknown)'}`")
    sections.append(f"- **provenance_digest**: `{_field(manifest, 'provenance_digest') or '(unknown)'}`")
    sections.append("")

    hw = _field(manifest, "hardware")
    if hw is not None:
        sections.append("## Host hardware")
        sections.append("")
        for label, attr in [
            ("Chip", "chip_brand"),
            ("Family", "chip_family"),
            ("Variant", "chip_variant"),
            ("Apple generation", "apple_generation"),
            ("Model identifier", "model_identifier"),
            ("Performance cores", "cpu_cores_performance"),
            ("Efficiency cores", "cpu_cores_efficiency"),
            ("Logical cores", "cpu_cores_logical"),
            ("GPU cores", "gpu_core_count"),
            ("ANE present", "ane_present"),
            ("RAM (bytes)", "ram_bytes"),
            ("Host ID hash", "host_id_hash"),
        ]:
            sections.append(f"- **{label}**: `{_field(hw, attr)}`")
        sections.append("")

    os_info = _field(manifest, "os")
    if os_info is not None:
        sections.append("## Operating system")
        sections.append("")
        for label, attr in [
            ("Product", "product_name"),
            ("Version", "product_version"),
            ("Build", "build_version"),
            ("Kernel", "kernel_version"),
            ("Boot args", "boot_args"),
            ("Metal version", "metal_version"),
            ("SIP", "sip_status"),
            ("Rosetta", "rosetta"),
        ]:
            sections.append(f"- **{label}**: `{_field(os_info, attr)}`")
        sections.append("")

    toolchain = _field(manifest, "toolchain")
    if toolchain is not None:
        sections.append("## Toolchain")
        sections.append("")
        for label, attr in [
            ("Python", "python_version_full"),
            ("Python executable", "python_executable_path"),
            ("uv version", "uv_version"),
            ("Harness uv.lock SHA-256", "uv_lock_sha256"),
            ("Xcode CLT", "xcode_clt_version"),
            ("xcode-select path", "xcode_select_path"),
        ]:
            sections.append(f"- **{label}**: `{_field(toolchain, attr)}`")
        sections.append("")
        sections.append("### Backend repos")
        sections.append("")
        sections.append("| Distribution | Version | Git SHA | Editable | Install URL |")
        sections.append("| --- | --- | --- | --- | --- |")
        for name, repo in _as_mapping(_field(toolchain, "backend_repo_versions")).items():
            sections.append(
                "| {} | {} | {} | {} | {} |".format(
                    name,
                    _field(repo, "version") or "N/A",
                    (_field(repo, "git_sha") or "")[:12] or "N/A",
                    "yes" if _field(repo, "editable", False) else "no",
                    _field(repo, "install_url") or "N/A",
                )
            )
        sections.append("")

    repo = _field(manifest, "repo")
    if repo is not None:
        sections.append("## Repository state")
        sections.append("")
        sections.append(f"- **Branch**: `{_field(repo, 'branch')}`")
        sections.append(f"- **Commit**: `{_field(repo, 'git_sha')}`")
        sections.append(f"- **Describe**: `{_field(repo, 'describe')}`")
        sections.append(f"- **Upstream**: `{_field(repo, 'upstream_url')}`")
        sections.append(f"- **Dirty**: `{_field(repo, 'dirty', False)}`")
        dirty_files = _field(repo, "dirty_files", []) or []
        if dirty_files:
            sections.append("- **Dirty files**:")
            for f in dirty_files[:20]:
                sections.append(f"  - `{f}`")
            if len(dirty_files) > 20:
                sections.append(f"  - … and {len(dirty_files) - 20} more")
        sections.append("")

    determinism = _field(manifest, "determinism")
    if determinism is not None:
        sections.append("## Determinism inputs")
        sections.append("")
        for label, attr in [
            ("Seed", "seed"),
            ("RNG", "rng_kind"),
            ("Batch size", "batch_size"),
            ("Latent shape", "latent_shape"),
            ("Latent SHA-256", "latent_sha256"),
            ("Text embedding shape", "text_embedding_shape"),
            ("Text embedding SHA-256", "text_embedding_sha256"),
            ("Text embedding source", "text_embedding_source"),
            ("Timestep", "timestep"),
            ("Shared input path", "shared_input_path"),
            ("Shared input SHA-256", "shared_input_sha256"),
            ("Checkpoint SHA-256", "checkpoint_sha256"),
            ("Checkpoint path", "checkpoint_path"),
        ]:
            sections.append(f"- **{label}**: `{_field(determinism, attr)}`")
        sections.append("")

    conditions = _field(manifest, "conditions")
    if conditions is not None:
        sections.append("## Runtime conditions")
        sections.append("")
        sections.append(f"- **Started**: `{_field(conditions, 'started_at_iso')}`")
        sections.append(f"- **Finished**: `{_field(conditions, 'finished_at_iso')}`")
        sections.append(f"- **Wall duration (s)**: `{_field(conditions, 'wall_duration_s')}`")
        for label, attr in [
            ("Power at start", "power_at_start"),
            ("Power at end", "power_at_end"),
            ("Load at start", "load_at_start"),
            ("Load at end", "load_at_end"),
            ("Thermal at start", "thermal_at_start"),
            ("Thermal at end", "thermal_at_end"),
        ]:
            sections.append(f"- **{label}**: `{_field(conditions, attr)}`")
        sections.append("")

    sampler = _field(manifest, "power_sampler")
    if sampler is not None:
        sections.append("## Power sampler")
        sections.append("")
        for label, attr in [
            ("powermetrics version", "powermetrics_version"),
            ("Interval (ms)", "interval_ms"),
            ("Samplers", "samplers"),
            ("Baseline (s)", "baseline_seconds"),
            ("sudo cached", "sudo_cached"),
            ("Samples (total)", "sample_count_total"),
            ("Samples (baseline window)", "sample_count_baseline_window"),
            ("Plist path", "plist_path"),
            ("Plist SHA-256", "plist_sha256"),
        ]:
            sections.append(f"- **{label}**: `{_field(sampler, attr)}`")
        sections.append("")

    env_vars = _field(manifest, "env_vars")
    if env_vars is not None:
        values = _as_mapping(_field(env_vars, "values"))
        sections.append("## Environment variables (captured prefixes)")
        sections.append("")
        if not values:
            sections.append("- (no matching variables set)")
        else:
            for key, value in sorted(values.items()):
                sections.append(f"- `{key}={value}`")
        sections.append("")

    Path(path).write_text("\n".join(sections) + "\n", encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fmt_range(lo, hi) -> str:
    """Render a between-pass percentile range as ``p10–p90`` or ``N/A``.

    Empty when the session had too few passes for the aggregator to compute the
    interval (<3 valid samples); the lone median already lives in its own
    column so we don't double-print it."""
    if lo is None or hi is None:
        return "N/A"
    return f"{_fmt(lo)}–{_fmt(hi)}"


def _fmt_median_with_spread(median, lo, hi) -> str:
    """Render the energy / power columns as ``median [p10–p90]``.

    Inlining the spread next to the median keeps the table one row per cell
    while making the between-run uncertainty impossible to overlook — the whole
    point of multi-run mode. Falls back to plain median when the percentile
    bounds are unavailable (single-run rows, or sessions with <3 passes)."""
    if median is None:
        return "N/A"
    if lo is None or hi is None:
        return _fmt(median)
    return f"{_fmt(median)} [{_fmt(lo)}–{_fmt(hi)}]"


def _multi_run_caption(records) -> str:
    counts = sorted({record.repeat_count for record in records if record.repeat_count})
    if not counts:
        return "multi-run aggregate"
    counts_label = ",".join(str(c) for c in counts)
    return f"multi-run aggregate · repeats={counts_label}"


def _fmt_eq(value, spec: str) -> str:
    """Format an equivalence metric with an explicit spec.

    ``_fmt`` collapses to ``%g`` which would mask sub-µ MSE values and round
    a cosine of 0.9999996 to ``"1"``. The equivalence table needs to surface
    that drift, so callers pass the precision they want."""
    if value is None:
        return "N/A"
    return format(float(value), spec)


def _fmt_flag(flag) -> str:
    if flag is None:
        return "N/A"
    return "yes" if flag else "no"


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
