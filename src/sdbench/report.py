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
from dataclasses import dataclass
from pathlib import Path

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

    @property
    def ok(self) -> bool:
        return (
            self.schema_ok
            and self.digests_consistent
            and self.digests_match_manifest
            and self.latent_consistent
            and self.text_embedding_consistent
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

    matrix = Path(matrix_config) if matrix_config else workspace.matrix_path
    if matrix.exists():
        shutil.copy2(matrix, bundle_dir / "matrix.yaml")

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


def validate_report(bundle: Path | str) -> ValidationReport:
    """Validate a contributor bundle: schema version + digest + SHA consistency."""
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

    return ValidationReport(
        schema_version=schema_version if isinstance(schema_version, int) else None,
        supported_schema=TELEMETRY_SCHEMA_VERSION,
        schema_ok=schema_ok,
        digests_consistent=digests_consistent,
        digests_match_manifest=digests_match_manifest,
        latent_consistent=latent_consistent,
        text_embedding_consistent=text_embedding_consistent,
        issues=issues,
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
        "- `matrix.yaml` — the matrix config the run consumed.\n"
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
