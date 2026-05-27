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


def collect_environment_manifest(
    seed: int,
    run_conditions: str,
    checkpoint_path: str | Path | None = None,
) -> EnvironmentManifest:
    packages = {}
    for name in ["numpy", "torch", "coremltools", "mlx", "diffusers", "psutil"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    resolved_checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path else None
    return EnvironmentManifest(
        chip_model=platform.processor() or platform.machine(),
        os_version=platform.platform(),
        python_version=platform.python_version(),
        package_versions=packages,
        checkpoint_path=str(resolved_checkpoint) if resolved_checkpoint else None,
        checkpoint_sha256=_sha256(resolved_checkpoint) if resolved_checkpoint and resolved_checkpoint.is_file() else None,
        seed=seed,
        run_conditions=run_conditions,
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
