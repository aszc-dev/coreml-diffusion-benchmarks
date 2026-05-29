"""Run provenance: a content fingerprint that keeps every datapoint traceable.

The comparison is weight-sensitive (R0.2) and version-sensitive (R10.4, R11.3):
mixing results produced from different checkpoints or framework versions silently
corrupts the study. The fingerprint covers the checkpoint hash, the tool version,
the host chip, and the pinned dependency sets of all three uv environments
(harness, ct8 converter, ct9 converter). Results are stamped with its digest, and
a ledger records the current fingerprint so a changed input can invalidate stale
cached results before new ones are upserted.
"""

import hashlib
import json
import time
import tomllib
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

KEY_PACKAGES = ("torch", "coremltools", "diffusers", "mlx", "numpy", "transformers")
TOOL_DIST = "coreml-diffusion-benchmarks"


@dataclass(frozen=True)
class Fingerprint:
    checkpoint_sha256: str | None
    tool_version: str
    chip: str
    harness_deps: dict[str, str]
    convert_ct8_deps: dict[str, str]
    convert_ct9_deps: dict[str, str]
    # Per-host identity hash (salted sysctl kern.uuid) and per-environment lock
    # content hashes — so a different machine OR a re-resolved lock yields a
    # different digest and the stale-results sweep fires (R11.6, R11.9).
    # Defaulted for back-compat with older callers/tests; the live collector
    # always populates them.
    host_id_hash: str | None = None
    harness_uv_lock_sha256: str | None = None
    convert_ct8_uv_lock_sha256: str | None = None
    convert_ct9_uv_lock_sha256: str | None = None

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Ledger:
    digest: str
    fingerprint: dict
    run_ids: list[str]
    updated_s: float


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_lock_versions(lock_path: str | Path, packages: tuple[str, ...] = KEY_PACKAGES) -> dict[str, str]:
    """Pinned versions of the key frameworks in a uv.lock, or {} if the lock is absent."""
    path = Path(lock_path)
    if not path.exists():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    versions = {p["name"]: p.get("version") for p in data.get("package", [])}
    return {name: versions[name] for name in packages if versions.get(name)}


def tool_version() -> str:
    try:
        return metadata.version(TOOL_DIST)
    except metadata.PackageNotFoundError:
        return "0.0.0+unknown"


def collect_fingerprint(ws, checkpoint_sha256: str | None, chip: str | None = None) -> Fingerprint:
    if chip is None:
        from sdbench.tui.capabilities import detect_capabilities

        chip = detect_capabilities().chip
    from sdbench.telemetry import host_id_hash as _host_id_hash, sha256_file as _sha256_file

    return Fingerprint(
        checkpoint_sha256=checkpoint_sha256,
        tool_version=tool_version(),
        chip=chip,
        harness_deps=parse_lock_versions(ws.root / "uv.lock"),
        convert_ct8_deps=parse_lock_versions(ws.root / "envs" / "apple-ct8" / "uv.lock"),
        convert_ct9_deps=parse_lock_versions(ws.root / "envs" / "team-ct9" / "uv.lock"),
        host_id_hash=_host_id_hash() or None,
        harness_uv_lock_sha256=_sha256_file(ws.root / "uv.lock"),
        convert_ct8_uv_lock_sha256=_sha256_file(ws.root / "envs" / "apple-ct8" / "uv.lock"),
        convert_ct9_uv_lock_sha256=_sha256_file(ws.root / "envs" / "team-ct9" / "uv.lock"),
    )


def load_ledger(path: str | Path) -> Ledger | None:
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return Ledger(
        digest=data["digest"],
        fingerprint=data.get("fingerprint", {}),
        run_ids=list(data.get("run_ids", [])),
        updated_s=float(data.get("updated_s", 0.0)),
    )


def save_ledger(ledger: Ledger, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(ledger), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_stale(current: Fingerprint, ledger: Ledger | None) -> bool:
    """True when a prior ledger exists and its fingerprint differs from the current one."""
    return ledger is not None and ledger.digest != current.digest


def record_run(ws, current: Fingerprint, run_id: str) -> Ledger:
    """Record a run under the current fingerprint, resetting the run list if the digest changed."""
    existing = load_ledger(ws.provenance_path)
    run_ids = list(existing.run_ids) if (existing and existing.digest == current.digest) else []
    if run_id not in run_ids:
        run_ids.append(run_id)
    ledger = Ledger(digest=current.digest, fingerprint=asdict(current), run_ids=run_ids, updated_s=time.time())
    save_ledger(ledger, ws.provenance_path)
    return ledger


@dataclass(frozen=True)
class VerifyReport:
    total: int
    digests: list[str]
    current_digest: str
    consistent: bool       # every stamped record shares one digest
    matches_current: bool  # ...and that digest is the current environment's

    @property
    def ok(self) -> bool:
        return self.consistent and self.matches_current


def verify_results(records, current: Fingerprint) -> VerifyReport:
    """Check that all results were produced under one provenance, equal to the current one.

    Surfaces accidentally-mixed datapoints (different checkpoint or framework
    versions) — the misuse the fingerprint exists to prevent.
    """
    digests = sorted({r.provenance_digest for r in records if r.provenance_digest})
    return VerifyReport(
        total=len(records),
        digests=digests,
        current_digest=current.digest,
        consistent=len(digests) <= 1,
        matches_current=digests == [current.digest] if digests else False,
    )


def invalidate_stale_results(ws, current: Fingerprint) -> list[str]:
    """If the stored fingerprint differs, remove dependent result files so the next run is clean.

    Converted artifacts are not auto-deleted here (expensive, 12 GB+); they are
    reconciled against the checkpoint hash by the conversion orchestrator.
    """
    if not is_stale(current, load_ledger(ws.provenance_path)):
        return []
    removed: list[str] = []
    results = ws.results_data_dir / "results.jsonl"
    if results.exists():
        results.unlink()
        removed.append(str(results))
    if ws.results_tables_dir.exists():
        for table in sorted(ws.results_tables_dir.glob("*.md")):
            table.unlink()
            removed.append(str(table))
    return removed
