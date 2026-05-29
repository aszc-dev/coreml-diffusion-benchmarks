"""Per-run environment manifest: every fact a contributor needs to re-run the benchmark.

Built on top of :mod:`sdbench.telemetry` — this module is the assembly layer that
glues the individual probes into one frozen dataclass, written as
``results/data/environment.json`` (latest) plus a per-run ledger copy under
``results/data/environments/<run_id>.json`` so distinct runs are never lost.

Legacy fields (``chip_model``, ``os_version``, ``package_versions``) are kept for
one release window so older external readers don't crash on the new schema; they
are mirrored from the structured telemetry blocks at write time.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sdbench import telemetry
from sdbench.inputs import SharedInput
from sdbench.telemetry import (
    ConversionToolchain,
    DeterminismInputs,
    HostEnvVars,
    HostHardware,
    HostOS,
    PowerSamplerMeta,
    RepoState,
    RunConditions,
    TELEMETRY_SCHEMA_VERSION,
    ToolchainVersions,
)


@dataclass(frozen=True)
class EnvironmentManifest:
    schema_version: int
    tool_version: str | None
    run_id: str | None
    provenance_digest: str | None
    hardware: HostHardware
    os: HostOS
    toolchain: ToolchainVersions
    repo: RepoState
    conversion_envs: dict[str, ConversionToolchain]
    determinism: DeterminismInputs | None
    env_vars: HostEnvVars
    power_sampler: PowerSamplerMeta | None
    conditions: RunConditions | None
    cells_run: list[str] = field(default_factory=list)
    probe_errors: dict[str, str] = field(default_factory=dict)

    # ---- Legacy mirror fields, removed at schema v3 ----------------------
    # Populated from the structured blocks above so external readers that still
    # parse the v1 manifest keep working until they upgrade.
    chip_model: str | None = None
    os_version: str | None = None
    python_version: str | None = None
    package_versions: dict[str, str] | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    seed: int | None = None
    run_conditions: str | None = None
    conversion_env_versions: dict[str, dict[str, str]] | None = None


def collect_environment_manifest(
    seed: int,
    run_conditions: str,
    *,
    checkpoint_path: str | Path | None = None,
    workspace=None,
    provenance_digest: str | None = None,
    run_id: str | None = None,
    shared_input: SharedInput | None = None,
    shared_input_path: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    power_sampler: PowerSamplerMeta | None = None,
    conditions: RunConditions | None = None,
    cells_run: list[str] | None = None,
) -> EnvironmentManifest:
    """Collect every reproducibility fact this run can see.

    ``workspace`` is optional so callers without a Workspace handle (legacy
    headless ``run-matrix``) still get a useful manifest, just without repo/lock
    hashes."""
    runner = None  # use telemetry's default subprocess.run with a 5s timeout

    hardware = telemetry.collect_host_hardware(runner)
    os_info = telemetry.collect_host_os(runner)
    repo_root = Path(workspace.root) if workspace is not None else Path.cwd()
    toolchain = telemetry.collect_toolchain_versions(repo_root, runner)
    conversion_envs = telemetry.collect_conversion_envs(repo_root)
    repo = telemetry.collect_repo_state(repo_root, runner)
    env_vars = telemetry.collect_env_vars()
    determinism: DeterminismInputs | None = None
    if shared_input is not None:
        determinism = telemetry.collect_determinism_inputs(
            seed=seed,
            batch_size=int(shared_input.latent.shape[0]),
            shared_input=shared_input,
            shared_input_path=shared_input_path,
            timestep=shared_input.timestep,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
        )

    tool_ver = None
    legacy_convert_env_versions: dict[str, dict[str, str]] | None = None
    if workspace is not None:
        from sdbench.provenance import parse_lock_versions, tool_version

        tool_ver = tool_version()
        legacy_convert_env_versions = {
            "apple_ct8": parse_lock_versions(workspace.root / "envs" / "apple-ct8" / "uv.lock"),
            "team_ct9": parse_lock_versions(workspace.root / "envs" / "team-ct9" / "uv.lock"),
        }

    # Legacy mirror values for readers still on the v1 schema.
    chip_model = hardware.chip_brand
    os_version = f"{os_info.product_name} {os_info.product_version} ({os_info.build_version})".strip()
    python_version = toolchain.python_version_full.split(" ")[0] if toolchain.python_version_full else None
    legacy_packages = {
        name: toolchain.harness_packages[name]
        for name in ("numpy", "torch", "coremltools", "mlx", "diffusers", "psutil")
        if name in toolchain.harness_packages
    }

    return EnvironmentManifest(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        tool_version=tool_ver,
        run_id=run_id,
        provenance_digest=provenance_digest,
        hardware=hardware,
        os=os_info,
        toolchain=toolchain,
        repo=repo,
        conversion_envs=conversion_envs,
        determinism=determinism,
        env_vars=env_vars,
        power_sampler=power_sampler,
        conditions=conditions,
        cells_run=list(cells_run or []),
        probe_errors={},
        chip_model=chip_model,
        os_version=os_version,
        python_version=python_version,
        package_versions=legacy_packages or None,
        checkpoint_path=str(Path(checkpoint_path).expanduser()) if checkpoint_path else None,
        checkpoint_sha256=checkpoint_sha256,
        seed=seed,
        run_conditions=run_conditions,
        conversion_env_versions=legacy_convert_env_versions,
    )


def write_environment_manifest(
    manifest: EnvironmentManifest,
    path: str | Path,
    *,
    history_dir: str | Path | None = None,
) -> None:
    """Write the manifest to ``path`` (latest) and, if ``history_dir`` is given,
    also keep a per-run copy under ``history_dir/<run_id>.json`` so a long-running
    ledger of distinct runs survives across reruns."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(manifest), indent=2, sort_keys=True, default=_json_default) + "\n"
    output.write_text(payload, encoding="utf-8")
    if history_dir and manifest.run_id:
        hist = Path(history_dir)
        hist.mkdir(parents=True, exist_ok=True)
        (hist / f"{manifest.run_id}.json").write_text(payload, encoding="utf-8")


def _json_default(value):
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
