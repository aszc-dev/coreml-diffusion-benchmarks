"""Contributor submission bundling and maintainer-side validation.

A *report* is a self-contained directory (optionally zipped) that lets a
maintainer reproduce, audit, or aggregate a contributor's run on an identical
machine. It bundles the environment manifest, the JSONL results, the raw
powermetrics plist, the generated tables, and the matrix config the run used.

Anonymization is opt-in and preserves every field needed for reproducibility:
the checkpoint SHA, seed, latent/text-embedding hashes, and toolchain
fingerprints all stay; only filesystem-local PII (``checkpoint_path``,
``repo.upstream_url``, ``repo.dirty_files``) is stripped, and ``host_id_hash`` is
re-hashed with a contributor-supplied salt.
"""

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from sdbench.results import BenchmarkRecord, load_jsonl
from sdbench.telemetry import TELEMETRY_SCHEMA_VERSION


@dataclass(frozen=True)
class ValidationReport:
    schema_version: int | None
    supported_schema: int
    schema_ok: bool
    digests_consistent: bool
    digests_match_manifest: bool
    latent_consistent: bool
    text_embedding_consistent: bool
    issues: list[str]
    # Multi-run additions (schema v3). When the bundle contains a session
    # (sessions/<id>/aggregated.jsonl + session.json), these surface that the
    # aggregate is statistically sound: n_runs_ok must be >= the configured
    # floor, and energy's between-run (p90-p10)/median must stay under the
    # reproducibility threshold or the cell is flagged as noisy.
    session_id: str | None = None
    n_runs_ok_min: int | None = None
    energy_spread_max: float | None = None
    session_ok: bool = True
    # Cells in the bundle that ran despite ``enabled: false`` in the source
    # matrix. Surfaced informationally — the bundle stays valid because the
    # bundled matrix records the override — but the maintainer should see what
    # diverged from the committed source.
    matrix_overrides: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.schema_ok
            and self.digests_consistent
            and self.digests_match_manifest
            and self.latent_consistent
            and self.text_embedding_consistent
            and self.session_ok
            and not self.issues
        )


def build_report(
    workspace,
    *,
    run_id: str | None = None,
    output_root: Path | str | None = None,
    matrix_config: Path | str | None = None,
    zip_bundle: bool = True,
    anonymize: bool = False,
    salt: str | None = None,
) -> Path:
    """Materialize a contributor submission bundle and return its path.

    When ``run_id`` is omitted, the manifest's own ``run_id`` is used; if that's
    missing too, "latest" is used as the bundle key. ``zip_bundle=True`` also
    writes a ``.zip`` alongside the directory."""
    if anonymize and not salt:
        raise ValueError("--salt is required when --anonymize is set; reuse it across runs to keep the host hash stable.")

    manifest_path = workspace.results_data_dir / "environment.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"environment manifest not found at {manifest_path}; run the benchmark first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_run_id = run_id or manifest.get("run_id") or "latest"

    bundle_root = Path(output_root) if output_root else workspace.results_dir / "reports"
    bundle_dir = bundle_root / resolved_run_id
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if anonymize:
        manifest = _anonymize_manifest(manifest, salt or "")

    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results_path = workspace.results_data_dir / "results.jsonl"
    if results_path.exists():
        records = load_jsonl(results_path)
        if anonymize:
            records = [_anonymize_record(record) for record in records]
        out_lines = []
        for record in records:
            payload = {
                key: value
                for key, value in record.__dict__.items()
                if not key.startswith("_")
            }
            out_lines.append(json.dumps(payload, sort_keys=True))
        (bundle_dir / "results.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    tables_dir = workspace.results_tables_dir
    if tables_dir.exists():
        shutil.copytree(tables_dir, bundle_dir / "tables", dirs_exist_ok=True)

    raw_dir = workspace.results_raw_dir
    if raw_dir.exists():
        bundle_raw = bundle_dir / "raw"
        bundle_raw.mkdir(exist_ok=True)
        for source in raw_dir.glob(f"{resolved_run_id}*"):
            shutil.copy2(source, bundle_raw / source.name)
        # If no per-run files were matched, copy the latest plist as a courtesy.
        if not any(bundle_raw.iterdir()):
            latest_plist = sorted(raw_dir.glob("*powermetrics.plist"))
            if latest_plist:
                shutil.copy2(latest_plist[-1], bundle_raw / latest_plist[-1].name)

    # Bundle the whole session directory when this run was part of one. The
    # aggregator's ``sessions/<id>/aggregated.jsonl`` is what the contributor
    # README points at as the headline result; the per-pass ``run-NN.jsonl``s
    # and ``session.json`` are kept alongside so a maintainer can audit the
    # raw passes the aggregate was computed from.
    session_id = manifest.get("session_id")
    if session_id:
        session_src = workspace.results_data_dir / "sessions" / session_id
        if session_src.exists():
            shutil.copytree(session_src, bundle_dir / "sessions" / session_id, dirs_exist_ok=True)

    matrix = Path(matrix_config) if matrix_config else workspace.matrix_path
    if matrix.exists():
        # Snapshot the matrix the run *actually executed*, not the raw source
        # file. The static yaml can carry ``enabled: false`` rows and ad-hoc
        # CLI selections diverge from it too; bundling it verbatim has misled
        # reviewers into thinking disabled cells were measured. The cells_run
        # list in the manifest is the source of truth for execution scope, so
        # we filter the yaml's ``cells:`` to that set and stamp a header that
        # makes the provenance explicit. The original is preserved alongside.
        executed_ids = list(manifest.get("cells_run") or [])
        override_ids = list(manifest.get("matrix_overrides") or [])
        _write_executed_matrix(matrix, bundle_dir / "matrix.yaml", executed_ids, override_ids)
        shutil.copy2(matrix, bundle_dir / "matrix.source.yaml")

    (bundle_dir / "README.md").write_text(_bundle_readme(manifest, anonymize), encoding="utf-8")
    (bundle_dir / "SCHEMA.md").write_text(_schema_doc(manifest.get("schema_version")), encoding="utf-8")

    if zip_bundle:
        zip_path = bundle_root / f"{resolved_run_id}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(bundle_dir.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(bundle_root))
        return zip_path
    return bundle_dir


#: Minimum ``n_runs_ok`` across cells for a multi-run aggregate to be accepted.
#: Three is the floor at which the p10/p90 interpolation in
#: :mod:`sdbench.aggregate` is well-defined and the median is no longer
#: dominated by a single pass.
SESSION_MIN_RUNS_OK = 3

#: Maximum tolerated ``(p90 - p10) / median`` on ``energy_per_unet_step_j``
#: for a multi-run aggregate to be accepted. Above this the cell's energy
#: figure is dominated by between-run noise (background processes, residual
#: thermal state) and any conclusion drawn from a single median is misleading.
#: 0.30 matches what we saw on the contaminated ``ours-ane-fp16`` and
#: ``mlx-gpu-fp16`` pairs that motivated this mode.
SESSION_MAX_ENERGY_SPREAD = 0.30


def validate_report(bundle: Path | str) -> ValidationReport:
    """Validate a contributor bundle: schema version + digest + SHA consistency.

    For multi-run bundles (``manifest.session_id`` present, plus a
    ``sessions/<id>/aggregated.jsonl`` next to the per-pass JSONLs) additionally
    enforces that every cell aggregated at least :data:`SESSION_MIN_RUNS_OK`
    passes and that the energy/step between-run spread stays within
    :data:`SESSION_MAX_ENERGY_SPREAD` (otherwise the aggregate's headline
    median is a coin flip)."""
    root = Path(bundle)
    issues: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        issues.append(f"manifest.json missing in {root}")
        return ValidationReport(
            schema_version=None,
            supported_schema=TELEMETRY_SCHEMA_VERSION,
            schema_ok=False,
            digests_consistent=False,
            digests_match_manifest=False,
            latent_consistent=False,
            text_embedding_consistent=False,
            issues=issues,
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, int):
        issues.append("manifest.schema_version is missing or not an integer")
    schema_ok = isinstance(schema_version, int) and schema_version <= TELEMETRY_SCHEMA_VERSION
    if isinstance(schema_version, int) and schema_version > TELEMETRY_SCHEMA_VERSION:
        issues.append(
            f"manifest schema_version={schema_version} is newer than supported {TELEMETRY_SCHEMA_VERSION}"
        )

    results_path = root / "results.jsonl"
    digests_consistent = True
    digests_match_manifest = True
    latent_consistent = True
    text_embedding_consistent = True
    manifest_digest = manifest.get("provenance_digest")
    determinism = manifest.get("determinism") or {}
    manifest_latent_sha = determinism.get("latent_sha256")
    manifest_text_sha = determinism.get("text_embedding_sha256")

    if results_path.exists():
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        digests = {row.get("provenance_digest") for row in rows if row.get("provenance_digest")}
        if len(digests) > 1:
            digests_consistent = False
            issues.append(f"results.jsonl carries {len(digests)} distinct provenance_digest values")
        if manifest_digest and digests and digests != {manifest_digest}:
            digests_match_manifest = False
            issues.append("provenance_digest in records does not match manifest")
        latents = {row.get("latent_input_sha256") for row in rows if row.get("latent_input_sha256")}
        if len(latents) > 1:
            latent_consistent = False
            issues.append(f"results.jsonl carries {len(latents)} distinct latent_input_sha256 values")
        if manifest_latent_sha and latents and latents != {manifest_latent_sha}:
            latent_consistent = False
            issues.append("latent_input_sha256 in records does not match manifest.determinism")
        texts = {row.get("text_embedding_input_sha256") for row in rows if row.get("text_embedding_input_sha256")}
        if len(texts) > 1:
            text_embedding_consistent = False
            issues.append(f"results.jsonl carries {len(texts)} distinct text_embedding_input_sha256 values")
        if manifest_text_sha and texts and texts != {manifest_text_sha}:
            text_embedding_consistent = False
            issues.append("text_embedding_input_sha256 in records does not match manifest.determinism")
    else:
        issues.append("results.jsonl missing")

    # Cells run despite ``enabled: false`` in the source matrix. Not a
    # validation failure — the bundled ``matrix.yaml`` carries them with
    # ``_matrix_override: true`` so cloning the bundle reproduces the same
    # scope — but surfaced on the report so a maintainer eyeballing it knows
    # the realised matrix diverges from the committed source.
    matrix_overrides = list(manifest.get("matrix_overrides") or [])

    session_id = manifest.get("session_id")
    n_runs_ok_min: int | None = None
    energy_spread_max: float | None = None
    session_ok = True
    if session_id:
        aggregated_path = root / "sessions" / session_id / "aggregated.jsonl"
        if not aggregated_path.exists():
            issues.append(f"sessions/{session_id}/aggregated.jsonl missing for multi-run bundle")
            session_ok = False
        else:
            agg_rows = [
                json.loads(line)
                for line in aggregated_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not agg_rows:
                issues.append("aggregated.jsonl is empty")
                session_ok = False
            # Cells reduced from too few ok passes have no business carrying a
            # p10/p90 — surface them as a hard fail so the bundle can't be
            # silently accepted as multi-run.
            n_ok_values = [int(row["n_runs_ok"]) for row in agg_rows if row.get("n_runs_ok") is not None]
            if n_ok_values:
                n_runs_ok_min = min(n_ok_values)
                if n_runs_ok_min < SESSION_MIN_RUNS_OK:
                    issues.append(
                        f"min n_runs_ok={n_runs_ok_min} below floor {SESSION_MIN_RUNS_OK} "
                        f"— rerun with --repeats >= {SESSION_MIN_RUNS_OK}"
                    )
                    session_ok = False
            # Energy spread: only computed when we actually have a p10/p90
            # (i.e. the cell aggregated >= 3 ok passes). Cells that failed too
            # often to interpolate are already flagged above.
            spreads: list[float] = []
            noisy: list[tuple[str, float]] = []
            for row in agg_rows:
                med = row.get("energy_per_unet_step_j")
                lo = row.get("energy_per_unet_step_j_p10")
                hi = row.get("energy_per_unet_step_j_p90")
                if med and lo is not None and hi is not None and med > 0:
                    rel = (float(hi) - float(lo)) / float(med)
                    spreads.append(rel)
                    if rel > SESSION_MAX_ENERGY_SPREAD:
                        noisy.append((str(row.get("cell_id", "?")), rel))
            if spreads:
                energy_spread_max = max(spreads)
            for cell_id, rel in noisy:
                issues.append(
                    f"cell '{cell_id}' energy spread (p90-p10)/median={rel:.2f} "
                    f"exceeds {SESSION_MAX_ENERGY_SPREAD:.2f} — add repeats or quieten the host"
                )
            if noisy:
                session_ok = False

    return ValidationReport(
        schema_version=schema_version if isinstance(schema_version, int) else None,
        supported_schema=TELEMETRY_SCHEMA_VERSION,
        schema_ok=schema_ok,
        digests_consistent=digests_consistent,
        digests_match_manifest=digests_match_manifest,
        latent_consistent=latent_consistent,
        text_embedding_consistent=text_embedding_consistent,
        issues=issues,
        session_id=session_id,
        n_runs_ok_min=n_runs_ok_min,
        energy_spread_max=energy_spread_max,
        session_ok=session_ok,
        matrix_overrides=matrix_overrides,
    )


# ---------------------------------------------------------------------------
# Anonymization
# ---------------------------------------------------------------------------


def _anonymize_manifest(manifest: dict, salt: str) -> dict:
    cleaned = json.loads(json.dumps(manifest))  # deep copy
    determinism = cleaned.get("determinism") or {}
    determinism.pop("checkpoint_path", None)  # leaks $HOME
    cleaned["determinism"] = determinism
    cleaned.pop("checkpoint_path", None)
    repo = cleaned.get("repo") or {}
    repo["upstream_url"] = None
    repo["dirty_files"] = []
    cleaned["repo"] = repo
    hardware = cleaned.get("hardware") or {}
    original = hardware.get("host_id_hash") or ""
    if original:
        hardware["host_id_hash"] = hashlib.sha256((salt + original).encode("utf-8")).hexdigest()[:16]
    cleaned["hardware"] = hardware
    return cleaned


def _anonymize_record(record: BenchmarkRecord) -> BenchmarkRecord:
    from dataclasses import replace

    return replace(record, host_id_hash=None)


def _write_executed_matrix(
    source: Path,
    destination: Path,
    executed_ids: list[str],
    override_ids: list[str] | None = None,
) -> None:
    """Render a yaml that reflects the realised run.

    The cell ordering follows ``executed_ids`` so the bundled matrix matches
    the order of rows in ``results.jsonl``. ``enabled: true`` is forced on the
    surviving cells and a header comment points at the original source file.
    Top-level keys (checkpoint, seed, equivalence, power, …) are kept as-is.

    Cells listed in ``override_ids`` ran despite the source matrix marking
    them ``enabled: false``; the bundled cell is annotated with a
    ``_matrix_override`` flag and a comment block names them at the top so a
    reader cloning the bundle understands why the realised matrix has more
    cells than the source. Without this marker the bundle silently rewrites
    history — the exact reproducibility hole the matrix snapshot exists to
    close."""
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        # If the source is unparseable, fall back to copying it verbatim so we
        # never silently drop information the reviewer might still need.
        shutil.copy2(source, destination)
        return
    overrides = set(override_ids or [])
    cells_by_id = {cell["id"]: cell for cell in (loaded.get("cells") or []) if isinstance(cell, dict) and "id" in cell}
    realised: list[dict] = []
    for cid in executed_ids:
        cell = cells_by_id.get(cid)
        if cell is None:
            # The CLI can run cells that aren't in the static yaml (selected
            # by tuple); we still want them represented in the bundled matrix.
            realised.append({"id": cid, "enabled": True, "_synthesised": True})
            continue
        cell = dict(cell)
        cell["enabled"] = True
        if cid in overrides:
            cell["_matrix_override"] = True
        realised.append(cell)
    loaded["cells"] = realised
    header_lines = [
        "# AUTO-GENERATED for this run. The original matrix.yaml is preserved as",
        "# matrix.source.yaml. Cells here are filtered to manifest.cells_run and",
        "# match the rows in results.jsonl one-for-one (and in that order).",
    ]
    if overrides:
        header_lines.append("#")
        header_lines.append(
            "# matrix_overrides: cells run despite enabled: false in matrix.source.yaml."
        )
        for cid in executed_ids:
            if cid in overrides:
                header_lines.append(f"#   - {cid}")
    header = "\n".join(header_lines) + "\n"
    destination.write_text(header + yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")


def _bundle_readme(manifest: dict, anonymized: bool) -> str:
    schema = manifest.get("schema_version")
    digest = manifest.get("provenance_digest") or "(unknown)"
    chip = ((manifest.get("hardware") or {}).get("chip_brand")) or "(unknown chip)"
    build = ((manifest.get("os") or {}).get("build_version")) or "(unknown build)"
    return (
        "# sdbench contributor report\n"
        "\n"
        f"- **schema_version**: `{schema}`\n"
        f"- **chip**: `{chip}`\n"
        f"- **macOS build**: `{build}`\n"
        f"- **provenance_digest**: `{digest}`\n"
        f"- **anonymized**: `{anonymized}`\n"
        "\n"
        "## Contents\n"
        "\n"
        "- `manifest.json` — full environment manifest (host, OS, toolchain, repo, determinism, runtime conditions).\n"
        "- `results.jsonl` — one record per matrix cell.\n"
        "- `tables/` — Markdown tables (latency, power_energy, size_quantization, conversion_time, environment).\n"
        "- `raw/` — retained powermetrics plists for auditability (R10.3).\n"
        "- `matrix.yaml` — the realised matrix: filtered + ordered to match `results.jsonl`.\n"
        "- `matrix.source.yaml` — the unmodified source file the run was launched from.\n"
        "- `SCHEMA.md` — schema notes and compatibility window.\n"
        "\n"
        "## Verification\n"
        "\n"
        "Run `uv run sdbench validate-report <this-directory>` from a clone of the harness at "
        "or above this manifest's schema version. The validator checks:\n"
        "\n"
        "1. `schema_version` is supported.\n"
        "2. All records share one `provenance_digest` matching the manifest.\n"
        "3. `latent_input_sha256` / `text_embedding_input_sha256` are consistent with `manifest.determinism`.\n"
    )


def _schema_doc(schema_version) -> str:
    return (
        "# Telemetry schema\n"
        "\n"
        f"- **bundle schema_version**: `{schema_version}`\n"
        f"- **maintainer supports up to**: `{TELEMETRY_SCHEMA_VERSION}`\n"
        "\n"
        "Additive field changes do not bump the version; breaking renames or removals do. "
        "A bundle with `schema_version > maintainer supported` will be refused by `validate-report` — "
        "upgrade the harness or downgrade the bundle.\n"
    )
