import json
import platform
import hashlib
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentManifest:
    chip_model: str
    os_version: str
    python_version: str
    package_versions: dict[str, str]
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    seed: int
    run_conditions: str
    # Pinned versions of the isolated conversion environments and the tool itself,
    # so the version-sensitive comparison is fully recorded (R10.4, R11.3). Defaulted
    # for back-compat with older manifests/callers.
    tool_version: str | None = None
    conversion_env_versions: dict[str, dict[str, str]] | None = None
    provenance_digest: str | None = None


def collect_environment_manifest(
    seed: int,
    run_conditions: str,
    checkpoint_path: str | Path | None = None,
    workspace=None,
    provenance_digest: str | None = None,
) -> EnvironmentManifest:
    packages = {}
    for name in ["numpy", "torch", "coremltools", "mlx", "diffusers", "psutil"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    resolved_checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path else None

    tool_version = None
    conversion_env_versions = None
    if workspace is not None:
        from sdbench.provenance import parse_lock_versions, tool_version as _tool_version

        tool_version = _tool_version()
        conversion_env_versions = {
            "apple_ct8": parse_lock_versions(workspace.root / "envs" / "apple-ct8" / "uv.lock"),
            "team_ct9": parse_lock_versions(workspace.root / "envs" / "team-ct9" / "uv.lock"),
        }

    return EnvironmentManifest(
        chip_model=platform.processor() or platform.machine(),
        os_version=platform.platform(),
        python_version=platform.python_version(),
        package_versions=packages,
        checkpoint_path=str(resolved_checkpoint) if resolved_checkpoint else None,
        checkpoint_sha256=_sha256(resolved_checkpoint) if resolved_checkpoint and resolved_checkpoint.is_file() else None,
        seed=seed,
        run_conditions=run_conditions,
        tool_version=tool_version,
        conversion_env_versions=conversion_env_versions,
        provenance_digest=provenance_digest,
    )


def write_environment_manifest(manifest: EnvironmentManifest, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
