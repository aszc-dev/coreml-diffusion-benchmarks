"""Reproducibility telemetry: every fact a contributor's run needs to be re-runnable.

This module is the single source of host/OS/toolchain/repo/runtime/determinism
probes used by the environment manifest (R10.4, R11.3) and by the per-cell record
(R11.6-R11.13). Every probe shells out with a short timeout and surfaces failures
as ``probe_errors`` entries rather than crashing the benchmark; a stuck ``pmset``
must never kill a 30-minute run. All collectors take an injectable ``Runner`` so
tests pass fakes — they never need a Mac, sudo, or live subprocesses.

The schema is versioned via :data:`TELEMETRY_SCHEMA_VERSION`; additive field
changes do not bump it, breaking renames or removals MUST (golden rule 13).
"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable

TELEMETRY_SCHEMA_VERSION = 2

# Hostname/UUID is hashed with a fixed salt so the published `host_id_hash` is
# usable for dedup across runs without leaking the underlying hardware UUID.
HOST_ID_SALT = "sdbench-host-v1"

# Behavior-affecting env var prefixes captured AFTER cli.py's module-level overrides
# apply, so the values reflect what each backend actually saw (R11.13).
ENV_VAR_PREFIXES: tuple[str, ...] = (
    "PYTORCH_",
    "TORCH_",
    "COREML_",
    "MLX_",
    "MPS_",
    "OMP_",
    "MKL_",
    "VECLIB_",
    "ACCELERATE_",
    "HF_",
    "HUGGINGFACE_",
    "TRANSFORMERS_",
    "DIFFUSERS_",
    "TOKENIZERS_",
    "TQDM_",
    "PYTHONHASHSEED",
    "PYTHONWARNINGS",
    "NUMEXPR_",
    "BLIS_",
)

# Backend distribution names whose installed version + (if editable/VCS) git SHA we
# want stamped into the toolchain block. Misses are recorded as None, not errors.
BACKEND_DISTRIBUTIONS: tuple[str, ...] = (
    "diffusers",
    "mlx",
    "coreml-diffusion",
    "python-coreml-stable-diffusion",
    "torch",
    "transformers",
    "coremltools",
)

Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _default_runner(argv: list[str]) -> "subprocess.CompletedProcess[str]":
    # Short timeout — a hung helper must never block the benchmark.
    return subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostHardware:
    chip_brand: str                       # "Apple M2 Pro" (sysctl machdep.cpu.brand_string)
    chip_family: str | None               # "M2"
    chip_variant: str | None              # "Pro" | "Max" | "Ultra" | "base" | None
    apple_generation: int | None          # 2
    model_identifier: str | None          # "Mac14,10" (sysctl hw.model)
    cpu_cores_performance: int | None     # sysctl hw.perflevel0.physicalcpu
    cpu_cores_efficiency: int | None      # sysctl hw.perflevel1.physicalcpu
    cpu_cores_logical: int | None         # sysctl hw.logicalcpu
    gpu_core_count: int | None            # system_profiler SPDisplaysDataType
    ane_present: bool                     # ioreg | grep '<class H<N>ANEIn'
    ram_bytes: int | None                 # sysctl hw.memsize
    host_id_hash: str                     # sha256(salt + kern.uuid)[:16]; "" if probe failed


@dataclass(frozen=True)
class HostOS:
    product_name: str                     # "macOS"
    product_version: str                  # "26.1"
    build_version: str                    # "25B62"
    kernel_version: str                   # uname -v / sysctl kern.version
    boot_args: str                        # nvram boot-args (empty if unset)
    metal_version: str | None             # /S/L/Frameworks/Metal.framework Info.plist CFBundleShortVersionString
    sip_status: str | None                # csrutil status (best-effort)
    rosetta: bool                         # sysctl sysctl.proc_translated == 1


@dataclass(frozen=True)
class HostPowerState:
    ac_powered: bool
    battery_percent: int | None
    low_power_mode: bool
    sleep_disabled: bool
    display_sleep_min: int | None
    caffeinate_pids: list[int]


@dataclass(frozen=True)
class HostLoad:
    loadavg_1m: float | None
    loadavg_5m: float | None
    loadavg_15m: float | None
    uptime_s: float | None
    process_count: int | None
    top_cpu_processes: list[dict[str, Any]]


@dataclass(frozen=True)
class ThermalSnapshot:
    cpu_speed_limit_pct: int | None
    throttled: bool
    source: str                           # "pmset" | "unavailable"
    detail: str


@dataclass(frozen=True)
class BackendRepoVersion:
    version: str | None
    git_sha: str | None
    install_url: str | None
    editable: bool


@dataclass(frozen=True)
class ToolchainVersions:
    python_implementation: str
    python_version_full: str
    python_executable_path: str
    uv_version: str | None
    uv_lock_sha256: str | None
    xcode_clt_version: str | None
    xcode_select_path: str | None
    harness_packages: dict[str, str]
    backend_repo_versions: dict[str, BackendRepoVersion]


@dataclass(frozen=True)
class ConversionToolchain:
    env_project: str                      # "envs/apple-ct8"
    uv_lock_sha256: str | None
    packages: dict[str, str]
    python_version: str | None
    driver_path: str | None
    driver_sha256: str | None


@dataclass(frozen=True)
class RepoState:
    git_sha: str | None
    branch: str | None
    dirty: bool
    dirty_files: list[str]
    upstream_url: str | None
    describe: str | None
    # Identifies the harness that actually ran when the workspace is not the
    # harness git clone (the common ``uv tool install`` deployment). Populated
    # from the wheel's stamped ``_build_info`` first, then PEP 610 direct_url,
    # then the workspace probe above. ``None`` only when the harness was loaded
    # from a source tree with no git, no build stamp, and no install metadata.
    harness_git_sha: str | None = None
    harness_git_describe: str | None = None
    harness_provenance_source: str | None = None  # "build_stamp"|"pep610"|"workspace"|"none"


@dataclass(frozen=True)
class PowerSamplerMeta:
    powermetrics_version: str | None
    interval_ms: int
    samplers: list[str]
    baseline_seconds: float
    sudo_cached: bool
    sample_count_total: int | None
    sample_count_baseline_window: int | None
    plist_path: str | None
    plist_sha256: str | None


@dataclass(frozen=True)
class DeterminismInputs:
    seed: int
    rng_kind: str                         # "numpy.default_rng (PCG64)"
    batch_size: int
    latent_shape: list[int]
    latent_sha256: str
    text_embedding_shape: list[int]
    text_embedding_sha256: str
    timestep: int
    shared_input_path: str | None
    shared_input_sha256: str | None
    text_embedding_source: str            # "random_normal" | "checkpoint_text_encoder"
    checkpoint_path: str | None
    checkpoint_sha256: str | None


@dataclass(frozen=True)
class HostEnvVars:
    prefixes_captured: list[str]
    values: dict[str, str]

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.values, sort_keys=True).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class RunConditions:
    note: str
    started_at_iso: str
    finished_at_iso: str | None
    wall_duration_s: float | None
    power_at_start: HostPowerState | None
    power_at_end: HostPowerState | None
    load_at_start: HostLoad | None
    load_at_end: HostLoad | None
    thermal_at_start: ThermalSnapshot | None
    thermal_at_end: ThermalSnapshot | None


# ---------------------------------------------------------------------------
# Shell-out helpers
# ---------------------------------------------------------------------------


def _run(runner: Runner, argv: list[str]) -> tuple[int, str, str]:
    rc, out, err = _run_raw(runner, argv)
    return (rc, out.strip(), err.strip())


def _run_raw(runner: Runner, argv: list[str]) -> tuple[int, str, str]:
    """Variant of :func:`_run` that preserves leading/trailing whitespace.

    Required for ``git status --porcelain`` parsing where the leading space in
    ``" M path"`` carries semantic status — stripping it shifts the path offset.
    """
    try:
        result = runner(argv)
    except (FileNotFoundError, subprocess.SubprocessError, OSError, subprocess.TimeoutExpired):
        return (-1, "", "probe-failed")
    return (result.returncode, result.stdout or "", result.stderr or "")


def _sysctl(runner: Runner, name: str) -> str | None:
    rc, out, _ = _run(runner, ["sysctl", "-n", name])
    return out if rc == 0 and out else None


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _int_or_none(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


def host_id_hash(runner: Runner | None = None) -> str:
    runner = runner or _default_runner
    uuid = _sysctl(runner, "kern.uuid") or ""
    if not uuid:
        return ""
    return hashlib.sha256((HOST_ID_SALT + uuid).encode("utf-8")).hexdigest()[:16]


_CHIP_RE = re.compile(r"Apple\s+M(\d+)(?:\s+(Pro|Max|Ultra))?", re.IGNORECASE)


def _parse_chip(brand: str) -> tuple[str | None, str | None, int | None]:
    match = _CHIP_RE.search(brand or "")
    if not match:
        return (None, None, None)
    generation = int(match.group(1))
    variant = (match.group(2) or "base").capitalize()
    return (f"M{generation}", variant, generation)


def collect_host_hardware(runner: Runner | None = None) -> HostHardware:
    runner = runner or _default_runner
    brand = _sysctl(runner, "machdep.cpu.brand_string") or platform.processor() or "unknown"
    family, variant, generation = _parse_chip(brand)
    gpu_cores = _probe_gpu_core_count(runner)
    ane_present = _probe_ane_present(runner)
    return HostHardware(
        chip_brand=brand,
        chip_family=family,
        chip_variant=variant,
        apple_generation=generation,
        model_identifier=_sysctl(runner, "hw.model"),
        cpu_cores_performance=_int_or_none(_sysctl(runner, "hw.perflevel0.physicalcpu")),
        cpu_cores_efficiency=_int_or_none(_sysctl(runner, "hw.perflevel1.physicalcpu")),
        cpu_cores_logical=_int_or_none(_sysctl(runner, "hw.logicalcpu")),
        gpu_core_count=gpu_cores,
        ane_present=ane_present,
        ram_bytes=_int_or_none(_sysctl(runner, "hw.memsize")),
        host_id_hash=host_id_hash(runner),
    )


def _probe_gpu_core_count(runner: Runner) -> int | None:
    rc, out, _ = _run(runner, ["system_profiler", "-json", "SPDisplaysDataType"])
    if rc != 0 or not out:
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    for entry in data.get("SPDisplaysDataType", []):
        cores = entry.get("sppci_cores") or entry.get("spdisplays_cores")
        n = _int_or_none(str(cores)) if cores is not None else None
        if n:
            return n
    return None


_ANE_CLASS_RE = re.compile(r"<class\s+H\d+ANEIn\b")


def _probe_ane_present(runner: Runner) -> bool:
    # The ANE registers as ``H<N>ANEIn`` (e.g. ``H11ANEIn`` on M2 Pro). The class
    # name lives several levels below ``AppleARMIODevice`` in the IORegistry, so
    # the previous ``-d 1`` query never matched anything on real hardware. ``-l``
    # walks the full tree once; the regex covers current and future ANE revisions.
    rc, out, _ = _run(runner, ["ioreg", "-l"])
    if rc != 0:
        return False
    return bool(_ANE_CLASS_RE.search(out))


def collect_host_os(runner: Runner | None = None) -> HostOS:
    runner = runner or _default_runner
    product_name = _sw_vers(runner, "-productName") or "macOS"
    product_version = _sw_vers(runner, "-productVersion") or platform.mac_ver()[0] or ""
    build = _sw_vers(runner, "-buildVersion") or ""
    kernel = _sysctl(runner, "kern.version") or ""
    boot_args = _nvram_boot_args(runner)
    rosetta = _sysctl(runner, "sysctl.proc_translated") == "1"
    return HostOS(
        product_name=product_name,
        product_version=product_version,
        build_version=build,
        kernel_version=_first_line(kernel),
        boot_args=boot_args,
        metal_version=_probe_metal_version(),
        sip_status=_probe_sip(runner),
        rosetta=rosetta,
    )


def _sw_vers(runner: Runner, flag: str) -> str | None:
    rc, out, _ = _run(runner, ["sw_vers", flag])
    return out if rc == 0 and out else None


def _nvram_boot_args(runner: Runner) -> str:
    rc, out, _ = _run(runner, ["nvram", "boot-args"])
    if rc != 0 or not out:
        return ""
    # nvram prints "boot-args\t<value>" — strip the key.
    parts = out.split("\t", 1)
    return parts[1].strip() if len(parts) == 2 else out


def _probe_metal_version() -> str | None:
    plist = Path("/System/Library/Frameworks/Metal.framework/Resources/Info.plist")
    if not plist.exists():
        return None
    try:
        import plistlib

        data = plistlib.loads(plist.read_bytes())
    except Exception:
        return None
    return data.get("CFBundleShortVersionString") or data.get("CFBundleVersion")


def _probe_sip(runner: Runner) -> str | None:
    rc, out, _ = _run(runner, ["csrutil", "status"])
    return _first_line(out) if rc == 0 and out else None


def collect_host_power_state(runner: Runner | None = None) -> HostPowerState:
    runner = runner or _default_runner
    rc, out, _ = _run(runner, ["pmset", "-g", "batt"])
    ac_powered = "AC Power" in out if rc == 0 else False
    battery = _parse_battery_pct(out) if rc == 0 else None

    rc2, custom, _ = _run(runner, ["pmset", "-g", "custom"])
    low_power = bool(re.search(r"lowpowermode\s+1", custom)) if rc2 == 0 else False
    sleep_disabled = bool(re.search(r"SleepDisabled\s+1", custom)) if rc2 == 0 else False
    display_sleep_match = re.search(r"displaysleep\s+(\d+)", custom) if rc2 == 0 else None
    display_sleep = int(display_sleep_match.group(1)) if display_sleep_match else None

    return HostPowerState(
        ac_powered=ac_powered,
        battery_percent=battery,
        low_power_mode=low_power,
        sleep_disabled=sleep_disabled,
        display_sleep_min=display_sleep,
        caffeinate_pids=_pgrep(runner, "caffeinate"),
    )


def _parse_battery_pct(text: str) -> int | None:
    match = re.search(r"(\d+)%", text)
    return int(match.group(1)) if match else None


def _pgrep(runner: Runner, name: str) -> list[int]:
    rc, out, _ = _run(runner, ["pgrep", "-x", name])
    if rc != 0 or not out:
        return []
    pids: list[int] = []
    for line in out.splitlines():
        pid = _int_or_none(line.strip())
        if pid is not None:
            pids.append(pid)
    return pids


def collect_host_load(runner: Runner | None = None) -> HostLoad:
    runner = runner or _default_runner
    try:
        load = os.getloadavg()
    except (OSError, AttributeError):
        load = (None, None, None)  # type: ignore[assignment]

    uptime = _uptime_seconds(runner)
    process_count = _process_count()
    top = _top_cpu_processes(runner)
    return HostLoad(
        loadavg_1m=load[0],
        loadavg_5m=load[1],
        loadavg_15m=load[2],
        uptime_s=uptime,
        process_count=process_count,
        top_cpu_processes=top,
    )


def _uptime_seconds(runner: Runner) -> float | None:
    boottime = _sysctl(runner, "kern.boottime")
    if not boottime:
        return None
    match = re.search(r"sec\s*=\s*(\d+)", boottime)
    if not match:
        return None
    return time.time() - int(match.group(1))


def _process_count() -> int | None:
    try:
        import psutil

        return len(psutil.pids())
    except Exception:
        return None


def _top_cpu_processes(runner: Runner, limit: int = 5) -> list[dict[str, Any]]:
    rc, out, _ = _run(runner, ["ps", "-Ao", "pid,pcpu,comm", "-r"])
    if rc != 0 or not out:
        return []
    rows: list[dict[str, Any]] = []
    for line in out.splitlines()[1 : limit + 1]:
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pid = _int_or_none(parts[0])
        try:
            cpu = float(parts[1])
        except ValueError:
            continue
        if pid is None:
            continue
        rows.append({"pid": pid, "cpu_pct": cpu, "name": parts[2]})
    return rows


def collect_thermal_snapshot(runner: Runner | None = None) -> ThermalSnapshot:
    """Sample CPU thermal pressure via ``pmset -g therm`` (R5.6)."""
    runner = runner or _default_runner
    rc, out, _ = _run(runner, ["pmset", "-g", "therm"])
    if rc != 0:
        return ThermalSnapshot(None, False, "unavailable", "pmset unavailable")
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    if not match:
        return ThermalSnapshot(None, False, "pmset", "no thermal pressure reported")
    limit = int(match.group(1))
    throttled = limit < 100
    return ThermalSnapshot(
        cpu_speed_limit_pct=limit,
        throttled=throttled,
        source="pmset",
        detail=f"CPU_Speed_Limit={limit}%" + (" (throttled)" if throttled else ""),
    )


def collect_repo_state(workspace_root: Path | str, runner: Runner | None = None) -> RepoState:
    runner = runner or _default_runner
    cwd_argv_prefix = ["git", "-C", str(workspace_root)]
    rc_sha, sha, _ = _run(runner, cwd_argv_prefix + ["rev-parse", "HEAD"])
    rc_branch, branch, _ = _run(runner, cwd_argv_prefix + ["rev-parse", "--abbrev-ref", "HEAD"])
    # `git status --porcelain` lines are `XY <path>` (two status chars + space + path).
    # Run with capture_output preserving leading whitespace so an unstaged change
    # (` M path`) keeps its leading space and the path offset stays at 3.
    rc_status, status, _ = _run_raw(runner, cwd_argv_prefix + ["status", "--porcelain"])
    rc_remote, remote, _ = _run(runner, cwd_argv_prefix + ["remote", "get-url", "origin"])
    rc_describe, describe, _ = _run(runner, cwd_argv_prefix + ["describe", "--always", "--dirty", "--tags"])

    dirty_files = (
        [line[3:].strip() for line in status.splitlines() if len(line) > 3][:200]
        if rc_status == 0
        else []
    )
    workspace_sha = sha if rc_sha == 0 and sha else None
    workspace_describe = describe if rc_describe == 0 and describe else None
    harness_sha, harness_describe, source = _resolve_harness_commit(workspace_sha, workspace_describe)
    return RepoState(
        git_sha=workspace_sha,
        branch=branch if rc_branch == 0 and branch else None,
        dirty=bool(dirty_files),
        dirty_files=dirty_files,
        upstream_url=remote if rc_remote == 0 and remote else None,
        describe=workspace_describe,
        harness_git_sha=harness_sha,
        harness_git_describe=harness_describe,
        harness_provenance_source=source,
    )


def _resolve_harness_commit(
    workspace_sha: str | None,
    workspace_describe: str | None,
) -> tuple[str | None, str | None, str]:
    """Pick the strongest available identifier for the harness commit that ran.

    Preference order:
        1. ``_build_info.BUILD_GIT_SHA`` stamped at wheel-build time.
        2. PEP 610 ``direct_url.json`` for ``coreml-diffusion-benchmarks`` —
           covers ``uv pip install git+…`` / ``pip install <url>`` installs.
        3. The workspace ``git rev-parse`` result (development from a clone).
    Returning the source lets the manifest reader explain a ``None`` SHA
    instead of leaving the field unattributed."""
    try:
        from sdbench import _build_info
    except ImportError:
        _build_info = None  # type: ignore[assignment]
    stamped_sha = getattr(_build_info, "BUILD_GIT_SHA", None) if _build_info else None
    stamped_describe = getattr(_build_info, "BUILD_GIT_DESCRIBE", None) if _build_info else None
    if stamped_sha:
        return stamped_sha, stamped_describe, "build_stamp"

    pep610_sha = _read_pep610_commit()
    if pep610_sha:
        return pep610_sha, None, "pep610"

    if workspace_sha:
        return workspace_sha, workspace_describe, "workspace"

    return None, None, "none"


def _read_pep610_commit() -> str | None:
    """Parse PEP 610 ``direct_url.json`` for the installed harness distribution."""
    try:
        dist = metadata.distribution("coreml-diffusion-benchmarks")
        raw = dist.read_text("direct_url.json")
    except (metadata.PackageNotFoundError, FileNotFoundError, OSError):
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    vcs = (data.get("vcs_info") or {}) if isinstance(data, dict) else {}
    commit = vcs.get("commit_id")
    return commit if isinstance(commit, str) and commit else None


def collect_env_vars(
    prefixes: Iterable[str] = ENV_VAR_PREFIXES,
    environ: dict[str, str] | None = None,
) -> HostEnvVars:
    env = environ if environ is not None else dict(os.environ)
    prefix_tuple = tuple(prefixes)
    values = {key: env[key] for key in sorted(env) if key.startswith(prefix_tuple)}
    return HostEnvVars(prefixes_captured=list(prefix_tuple), values=values)


def sha256_file(path: Path | str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_toolchain_versions(
    workspace_root: Path | str,
    runner: Runner | None = None,
) -> ToolchainVersions:
    runner = runner or _default_runner
    ws_root = Path(workspace_root)
    rc, uv_out, _ = _run(runner, ["uv", "--version"])
    uv_version = uv_out if rc == 0 and uv_out else None

    rc2, xcs_path, _ = _run(runner, ["xcode-select", "-p"])
    xcs = xcs_path if rc2 == 0 and xcs_path else None
    rc3, pkg_out, _ = _run(
        runner,
        ["pkgutil", "--pkg-info=com.apple.pkg.CLTools_Executables"],
    )
    xcode_clt = _parse_pkgutil_version(pkg_out) if rc3 == 0 else None

    return ToolchainVersions(
        python_implementation=platform.python_implementation(),
        python_version_full=sys.version.replace("\n", " "),
        python_executable_path=sys.executable,
        uv_version=uv_version,
        uv_lock_sha256=sha256_file(ws_root / "uv.lock"),
        xcode_clt_version=xcode_clt,
        xcode_select_path=xcs,
        harness_packages=_all_installed_packages(),
        backend_repo_versions=_backend_repo_versions(),
    )


def _parse_pkgutil_version(text: str) -> str | None:
    match = re.search(r"version:\s*(\S+)", text)
    return match.group(1) if match else None


def _all_installed_packages() -> dict[str, str]:
    out: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"] if dist.metadata else None
        if not name:
            continue
        out[name.lower()] = dist.version
    return out


def _backend_repo_versions() -> dict[str, BackendRepoVersion]:
    versions: dict[str, BackendRepoVersion] = {}
    for name in BACKEND_DISTRIBUTIONS:
        versions[name] = _read_backend_repo(name)
    return versions


def _read_backend_repo(name: str) -> BackendRepoVersion:
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return BackendRepoVersion(version=None, git_sha=None, install_url=None, editable=False)
    version = dist.version
    direct_url = None
    try:
        text = dist.read_text("direct_url.json")
        direct_url = json.loads(text) if text else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        direct_url = None
    git_sha = None
    install_url = None
    editable = False
    if isinstance(direct_url, dict):
        install_url = direct_url.get("url")
        vcs_info = direct_url.get("vcs_info") or {}
        git_sha = vcs_info.get("commit_id")
        editable = bool((direct_url.get("dir_info") or {}).get("editable"))
    return BackendRepoVersion(version=version, git_sha=git_sha, install_url=install_url, editable=editable)


def collect_conversion_envs(workspace_root: Path | str) -> dict[str, ConversionToolchain]:
    ws_root = Path(workspace_root)
    envs = {
        "apple_ct8": ws_root / "envs" / "apple-ct8",
        "team_ct9": ws_root / "envs" / "team-ct9",
    }
    drivers = {
        "apple_ct8": ws_root / "scripts" / "convert" / "apple_ct8.py",
        "team_ct9": ws_root / "scripts" / "convert" / "team_ct9.py",
    }
    out: dict[str, ConversionToolchain] = {}
    for key, env_root in envs.items():
        lock_path = env_root / "uv.lock"
        out[key] = ConversionToolchain(
            env_project=str(env_root.relative_to(ws_root)) if env_root.exists() else str(env_root),
            uv_lock_sha256=sha256_file(lock_path),
            packages=_parse_uv_lock_all(lock_path),
            python_version=_read_python_version(env_root),
            driver_path=str(drivers[key].relative_to(ws_root)) if drivers[key].exists() else None,
            driver_sha256=sha256_file(drivers[key]),
        )
    return out


def _parse_uv_lock_all(lock_path: Path) -> dict[str, str]:
    if not lock_path.exists():
        return {}
    try:
        data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    packages = data.get("package") or []
    return {p["name"]: p.get("version", "") for p in packages if p.get("name")}


def _read_python_version(env_root: Path) -> str | None:
    pin = env_root / ".python-version"
    if pin.is_file():
        return pin.read_text(encoding="utf-8").strip()
    pyproject = env_root / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return data.get("project", {}).get("requires-python")


def collect_power_sampler_meta(
    interval_ms: int,
    samplers: list[str],
    baseline_seconds: float,
    *,
    plist_path: Path | str | None,
    sudo_cached: bool,
    runner: Runner | None = None,
) -> PowerSamplerMeta:
    runner = runner or _default_runner
    return PowerSamplerMeta(
        powermetrics_version=_probe_powermetrics_fingerprint(runner),
        interval_ms=interval_ms,
        samplers=list(samplers),
        baseline_seconds=baseline_seconds,
        sudo_cached=sudo_cached,
        sample_count_total=None,
        sample_count_baseline_window=None,
        plist_path=str(plist_path) if plist_path else None,
        plist_sha256=sha256_file(plist_path) if plist_path else None,
    )


_POWERMETRICS_PATH = "/usr/bin/powermetrics"
_CODESIGN_SIGNED_TIME_RE = re.compile(r"Signed Time=([^\n]+)")


def _probe_powermetrics_fingerprint(runner: Runner) -> str | None:
    """Stable identifier for the powermetrics binary in use.

    ``powermetrics`` has no ``--version`` flag — the old probe captured the
    ``--help`` usage banner instead, producing ``"Usage: powermetrics ..."`` in
    the manifest. Apple ships the binary with macOS and re-signs it on every
    OS update, so we use ``codesign -dv``'s ``Signed Time`` line as a version
    proxy and pair it with the binary's mtime + size so two runs against the
    same binary yield the same string."""
    rc, out, err = _run(runner, ["codesign", "-dv", _POWERMETRICS_PATH])
    # codesign prints to stderr; combine streams so the regex sees the line.
    combined = (out or "") + "\n" + (err or "")
    parts: list[str] = []
    if rc == 0:
        match = _CODESIGN_SIGNED_TIME_RE.search(combined)
        if match:
            parts.append(f"signed={match.group(1).strip()}")
    try:
        stat = Path(_POWERMETRICS_PATH).stat()
        parts.append(f"size={stat.st_size}")
        parts.append(f"mtime={int(stat.st_mtime)}")
    except OSError:
        pass
    return "; ".join(parts) if parts else None


def collect_determinism_inputs(
    *,
    seed: int,
    batch_size: int,
    shared_input,
    shared_input_path: Path | str | None,
    timestep: int,
    checkpoint_path: Path | str | None,
    checkpoint_sha256: str | None,
    text_embedding_source: str = "random_normal",
    rng_kind: str = "numpy.default_rng(PCG64)",
) -> DeterminismInputs:
    return DeterminismInputs(
        seed=seed,
        rng_kind=rng_kind,
        batch_size=batch_size,
        latent_shape=list(shared_input.latent.shape),
        latent_sha256=hashlib.sha256(shared_input.latent.tobytes()).hexdigest(),
        text_embedding_shape=list(shared_input.text_embedding.shape),
        text_embedding_sha256=hashlib.sha256(shared_input.text_embedding.tobytes()).hexdigest(),
        timestep=timestep,
        shared_input_path=str(shared_input_path) if shared_input_path else None,
        shared_input_sha256=sha256_file(shared_input_path) if shared_input_path else None,
        text_embedding_source=text_embedding_source,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        checkpoint_sha256=checkpoint_sha256,
    )


# ---------------------------------------------------------------------------
# Top-level snapshot helpers used by run_cmd
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def snapshot_run_conditions_start(
    note: str,
    runner: Runner | None = None,
) -> RunConditions:
    return RunConditions(
        note=note,
        started_at_iso=now_iso(),
        finished_at_iso=None,
        wall_duration_s=None,
        power_at_start=collect_host_power_state(runner),
        power_at_end=None,
        load_at_start=collect_host_load(runner),
        load_at_end=None,
        thermal_at_start=collect_thermal_snapshot(runner),
        thermal_at_end=None,
    )


def snapshot_run_conditions_end(
    started: RunConditions,
    runner: Runner | None = None,
) -> RunConditions:
    finished_iso = now_iso()
    duration = _iso_delta_seconds(started.started_at_iso, finished_iso)
    return RunConditions(
        note=started.note,
        started_at_iso=started.started_at_iso,
        finished_at_iso=finished_iso,
        wall_duration_s=duration,
        power_at_start=started.power_at_start,
        power_at_end=collect_host_power_state(runner),
        load_at_start=started.load_at_start,
        load_at_end=collect_host_load(runner),
        thermal_at_start=started.thermal_at_start,
        thermal_at_end=collect_thermal_snapshot(runner),
    )


def _iso_delta_seconds(start_iso: str, end_iso: str) -> float | None:
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
    except ValueError:
        return None
    return (end - start).total_seconds()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def to_dict(obj: Any) -> Any:
    """Best-effort recursive ``asdict`` that also handles plain values."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(value) for value in obj]
    return obj
