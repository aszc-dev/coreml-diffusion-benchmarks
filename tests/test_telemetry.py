"""Tests for sdbench.telemetry probes.

Every shell-out is driven by an injected ``Runner`` callable so tests never need
sudo, a Mac, or live subprocesses. The fakes return canned CompletedProcess-like
objects keyed by argv prefix.
"""

import subprocess
from typing import Callable

from sdbench import inputs, telemetry


class FakeProc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def make_runner(responses: dict[tuple[str, ...], FakeProc]) -> Callable[[list[str]], FakeProc]:
    def runner(argv: list[str]) -> FakeProc:
        key = tuple(argv)
        for length in range(len(argv), 0, -1):
            prefix = tuple(argv[:length])
            if prefix in responses:
                return responses[prefix]
        return FakeProc(returncode=127, stdout="", stderr="not found")

    return runner


# ---------------------------------------------------------------------------
# host hardware
# ---------------------------------------------------------------------------


def test_collect_host_hardware_parses_sysctl_outputs():
    runner = make_runner(
        {
            ("sysctl", "-n", "machdep.cpu.brand_string"): FakeProc(0, "Apple M3 Max"),
            ("sysctl", "-n", "hw.model"): FakeProc(0, "Mac15,9"),
            ("sysctl", "-n", "hw.perflevel0.physicalcpu"): FakeProc(0, "12"),
            ("sysctl", "-n", "hw.perflevel1.physicalcpu"): FakeProc(0, "4"),
            ("sysctl", "-n", "hw.logicalcpu"): FakeProc(0, "16"),
            ("sysctl", "-n", "hw.memsize"): FakeProc(0, "137438953472"),
            ("sysctl", "-n", "kern.uuid"): FakeProc(0, "DEADBEEF-DEAD-DEAD-DEAD-DEADBEEFCAFE"),
            ("system_profiler", "-json", "SPDisplaysDataType"): FakeProc(
                0,
                '{"SPDisplaysDataType":[{"sppci_cores":"40"}]}',
            ),
            ("ioreg", "-l"): FakeProc(
                0, '          +-o H11ANE  <class H11ANEIn, id 0x100000505>'
            ),
        }
    )
    hw = telemetry.collect_host_hardware(runner)
    assert hw.chip_brand == "Apple M3 Max"
    assert hw.chip_family == "M3"
    assert hw.chip_variant == "Max"
    assert hw.apple_generation == 3
    assert hw.model_identifier == "Mac15,9"
    assert hw.cpu_cores_performance == 12
    assert hw.cpu_cores_efficiency == 4
    assert hw.cpu_cores_logical == 16
    assert hw.ram_bytes == 137438953472
    assert hw.gpu_core_count == 40
    assert hw.ane_present is True
    assert hw.host_id_hash and len(hw.host_id_hash) == 16


def test_host_id_hash_is_deterministic_and_salted():
    runner = make_runner({("sysctl", "-n", "kern.uuid"): FakeProc(0, "abc-uuid")})
    first = telemetry.host_id_hash(runner)
    second = telemetry.host_id_hash(runner)
    assert first == second
    assert "abc-uuid" not in first  # never leak the raw UUID


def test_host_id_hash_empty_when_probe_fails():
    runner = make_runner({})  # no responses -> rc=127
    assert telemetry.host_id_hash(runner) == ""


# ---------------------------------------------------------------------------
# OS
# ---------------------------------------------------------------------------


def test_collect_host_os_assembles_sw_vers_outputs():
    runner = make_runner(
        {
            ("sw_vers", "-productName"): FakeProc(0, "macOS"),
            ("sw_vers", "-productVersion"): FakeProc(0, "26.1"),
            ("sw_vers", "-buildVersion"): FakeProc(0, "25B62"),
            ("sysctl", "-n", "kern.version"): FakeProc(0, "Darwin Kernel Version 25.1.0: …"),
            ("nvram", "boot-args"): FakeProc(0, "boot-args\tdebug=0x100"),
            ("sysctl", "-n", "sysctl.proc_translated"): FakeProc(0, "0"),
            ("csrutil", "status"): FakeProc(0, "System Integrity Protection status: enabled."),
        }
    )
    os_info = telemetry.collect_host_os(runner)
    assert os_info.product_name == "macOS"
    assert os_info.build_version == "25B62"
    assert os_info.kernel_version.startswith("Darwin Kernel Version")
    assert os_info.boot_args == "debug=0x100"
    assert os_info.rosetta is False
    assert os_info.sip_status and "Integrity" in os_info.sip_status


def test_rosetta_flag_set_when_proc_translated_is_one():
    runner = make_runner(
        {
            ("sw_vers", "-productName"): FakeProc(0, "macOS"),
            ("sw_vers", "-productVersion"): FakeProc(0, "26.1"),
            ("sw_vers", "-buildVersion"): FakeProc(0, "25B62"),
            ("sysctl", "-n", "kern.version"): FakeProc(0, "k"),
            ("nvram", "boot-args"): FakeProc(1, ""),
            ("sysctl", "-n", "sysctl.proc_translated"): FakeProc(0, "1"),
        }
    )
    assert telemetry.collect_host_os(runner).rosetta is True


# ---------------------------------------------------------------------------
# repo
# ---------------------------------------------------------------------------


def test_collect_repo_state_parses_porcelain(tmp_path):
    porcelain = " M src/foo.py\n M src/bar.py\n?? new.txt\n"
    runner = make_runner(
        {
            ("git", "-C", str(tmp_path), "rev-parse", "HEAD"): FakeProc(0, "deadbeef"),
            ("git", "-C", str(tmp_path), "rev-parse", "--abbrev-ref", "HEAD"): FakeProc(0, "main"),
            ("git", "-C", str(tmp_path), "status", "--porcelain"): FakeProc(0, porcelain),
            ("git", "-C", str(tmp_path), "remote", "get-url", "origin"): FakeProc(0, "git@github.com:x/y"),
            ("git", "-C", str(tmp_path), "describe", "--always", "--dirty", "--tags"): FakeProc(0, "v0-dirty"),
        }
    )
    repo = telemetry.collect_repo_state(tmp_path, runner)
    assert repo.git_sha == "deadbeef"
    assert repo.branch == "main"
    assert repo.dirty is True
    assert repo.dirty_files == ["src/foo.py", "src/bar.py", "new.txt"]
    assert repo.upstream_url == "git@github.com:x/y"


def test_collect_repo_state_clean_repo():
    runner = make_runner(
        {
            ("git", "-C", ".", "rev-parse", "HEAD"): FakeProc(0, "abc"),
            ("git", "-C", ".", "rev-parse", "--abbrev-ref", "HEAD"): FakeProc(0, "main"),
            ("git", "-C", ".", "status", "--porcelain"): FakeProc(0, ""),
            ("git", "-C", ".", "remote", "get-url", "origin"): FakeProc(0, "u"),
            ("git", "-C", ".", "describe", "--always", "--dirty", "--tags"): FakeProc(0, "abc"),
        }
    )
    repo = telemetry.collect_repo_state(".", runner)
    assert repo.dirty is False
    assert repo.dirty_files == []


def test_repo_state_falls_back_to_harness_workspace_sha(monkeypatch):
    """When the workspace is a git clone of the harness itself, the workspace
    SHA doubles as the harness commit and ``harness_provenance_source`` says so."""
    runner = make_runner(
        {
            ("git", "-C", ".", "rev-parse", "HEAD"): FakeProc(0, "feedface"),
            ("git", "-C", ".", "rev-parse", "--abbrev-ref", "HEAD"): FakeProc(0, "main"),
            ("git", "-C", ".", "status", "--porcelain"): FakeProc(0, ""),
            ("git", "-C", ".", "remote", "get-url", "origin"): FakeProc(0, "u"),
            ("git", "-C", ".", "describe", "--always", "--dirty", "--tags"): FakeProc(0, "v0.1"),
        }
    )
    # Force the build-stamp + PEP 610 paths to miss so the workspace fallback runs.
    from sdbench import _build_info

    monkeypatch.setattr(_build_info, "BUILD_GIT_SHA", None, raising=False)
    monkeypatch.setattr(_build_info, "BUILD_GIT_DESCRIBE", None, raising=False)
    monkeypatch.setattr(telemetry, "_read_pep610_commit", lambda: None)

    repo = telemetry.collect_repo_state(".", runner)
    assert repo.harness_git_sha == "feedface"
    assert repo.harness_git_describe == "v0.1"
    assert repo.harness_provenance_source == "workspace"


def test_repo_state_prefers_build_stamp_over_workspace(monkeypatch):
    """``uv tool install`` runs from a non-repo workspace; the wheel's build
    stamp is the only honest harness identifier in that situation."""
    runner = make_runner({})  # every git call returns rc=127 → workspace fields stay None
    from sdbench import _build_info

    monkeypatch.setattr(_build_info, "BUILD_GIT_SHA", "cafebabe123", raising=False)
    monkeypatch.setattr(_build_info, "BUILD_GIT_DESCRIBE", "v0.1-2-gcafebab", raising=False)
    monkeypatch.setattr(telemetry, "_read_pep610_commit", lambda: None)

    repo = telemetry.collect_repo_state("/no/such/workspace", runner)
    assert repo.git_sha is None  # workspace probe failed (not a git clone)
    assert repo.harness_git_sha == "cafebabe123"
    assert repo.harness_git_describe == "v0.1-2-gcafebab"
    assert repo.harness_provenance_source == "build_stamp"


# ---------------------------------------------------------------------------
# thermal / load
# ---------------------------------------------------------------------------


def test_thermal_snapshot_throttled_when_limit_below_100():
    runner = make_runner(
        {("pmset", "-g", "therm"): FakeProc(0, "CPU_Speed_Limit \t = 65")}
    )
    snap = telemetry.collect_thermal_snapshot(runner)
    assert snap.throttled is True
    assert snap.cpu_speed_limit_pct == 65
    assert snap.source == "pmset"


def test_thermal_snapshot_unavailable_when_pmset_missing():
    runner = make_runner({})
    snap = telemetry.collect_thermal_snapshot(runner)
    assert snap.throttled is False
    assert snap.source == "unavailable"


def test_collect_host_power_state_reads_pmset():
    runner = make_runner(
        {
            ("pmset", "-g", "batt"): FakeProc(0, "Now drawing from 'AC Power'\n-InternalBattery-0 (id=12345) 95%"),
            ("pmset", "-g", "custom"): FakeProc(
                0,
                "AC Power:\n displaysleep         10\n SleepDisabled        1\n lowpowermode         0\n",
            ),
            ("pgrep", "-x", "caffeinate"): FakeProc(0, "1234\n5678"),
        }
    )
    state = telemetry.collect_host_power_state(runner)
    assert state.ac_powered is True
    assert state.battery_percent == 95
    assert state.sleep_disabled is True
    assert state.low_power_mode is False
    assert state.display_sleep_min == 10
    assert state.caffeinate_pids == [1234, 5678]


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_collect_determinism_inputs_hashes_arrays():
    shared = inputs.generate_shared_input(seed=7, resolution=64)
    det = telemetry.collect_determinism_inputs(
        seed=7,
        batch_size=int(shared.latent.shape[0]),
        shared_input=shared,
        shared_input_path=None,
        timestep=shared.timestep,
        checkpoint_path=None,
        checkpoint_sha256=None,
    )
    assert det.seed == 7
    assert det.rng_kind.startswith("numpy.default_rng")
    assert det.latent_shape == list(shared.latent.shape)
    assert len(det.latent_sha256) == 64
    assert len(det.text_embedding_sha256) == 64
    # Re-run with same seed → same hashes (determinism check).
    shared2 = inputs.generate_shared_input(seed=7, resolution=64)
    det2 = telemetry.collect_determinism_inputs(
        seed=7,
        batch_size=int(shared2.latent.shape[0]),
        shared_input=shared2,
        shared_input_path=None,
        timestep=shared2.timestep,
        checkpoint_path=None,
        checkpoint_sha256=None,
    )
    assert det.latent_sha256 == det2.latent_sha256
    assert det.text_embedding_sha256 == det2.text_embedding_sha256


def test_digest_shared_input_matches_telemetry_hash():
    shared = inputs.generate_shared_input(seed=11, resolution=64)
    digests = inputs.digest_shared_input(shared)
    det = telemetry.collect_determinism_inputs(
        seed=11,
        batch_size=int(shared.latent.shape[0]),
        shared_input=shared,
        shared_input_path=None,
        timestep=shared.timestep,
        checkpoint_path=None,
        checkpoint_sha256=None,
    )
    assert digests["latent"] == det.latent_sha256
    assert digests["text_embedding"] == det.text_embedding_sha256


def test_sha256_npz_file_roundtrip(tmp_path):
    shared = inputs.generate_shared_input(seed=3, resolution=64)
    path = tmp_path / "shared.npz"
    inputs.save_shared_input(shared, path)
    sha = inputs.sha256_npz_file(path)
    assert sha and len(sha) == 64
    assert inputs.sha256_npz_file(tmp_path / "missing.npz") is None


# ---------------------------------------------------------------------------
# env vars
# ---------------------------------------------------------------------------


def test_collect_env_vars_filters_by_prefix():
    env = {
        "PATH": "/usr/bin",
        "HOME": "/Users/x",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "COREML_NUM_THREADS": "4",
        "MLX_METAL_DEBUG": "0",
        "RANDOM": "noise",
    }
    out = telemetry.collect_env_vars(environ=env)
    assert set(out.values.keys()) == {
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "COREML_NUM_THREADS",
        "MLX_METAL_DEBUG",
    }
    assert "PATH" not in out.values
    assert out.digest and len(out.digest) == 64


# ---------------------------------------------------------------------------
# toolchain / conversion envs
# ---------------------------------------------------------------------------


def test_collect_toolchain_versions_hashes_uv_lock(tmp_path):
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "torch"\nversion = "2.6.0"\n', encoding="utf-8"
    )
    runner = make_runner(
        {
            ("uv", "--version"): FakeProc(0, "uv 0.5.0"),
            ("xcode-select", "-p"): FakeProc(0, "/Library/Developer/CommandLineTools"),
            ("pkgutil", "--pkg-info=com.apple.pkg.CLTools_Executables"): FakeProc(
                0, "package-id: com.apple.pkg.CLTools_Executables\nversion: 16.4\n"
            ),
        }
    )
    tc = telemetry.collect_toolchain_versions(tmp_path, runner)
    assert tc.uv_version == "uv 0.5.0"
    assert tc.uv_lock_sha256 and len(tc.uv_lock_sha256) == 64
    assert tc.xcode_clt_version == "16.4"
    assert tc.python_executable_path  # always populated


def test_collect_conversion_envs_reads_both_locks(tmp_path):
    for sub in ("apple-ct8", "team-ct9"):
        (tmp_path / "envs" / sub).mkdir(parents=True)
        (tmp_path / "envs" / sub / "uv.lock").write_text(
            f'[[package]]\nname = "coremltools"\nversion = "{sub}"\n', encoding="utf-8"
        )
    envs = telemetry.collect_conversion_envs(tmp_path)
    assert set(envs.keys()) == {"apple_ct8", "team_ct9"}
    assert envs["apple_ct8"].uv_lock_sha256 and len(envs["apple_ct8"].uv_lock_sha256) == 64
    assert envs["apple_ct8"].packages == {"coremltools": "apple-ct8"}


# ---------------------------------------------------------------------------
# probe error tolerance
# ---------------------------------------------------------------------------


def test_collectors_tolerate_missing_binaries():
    runner = make_runner({})  # everything 127
    hw = telemetry.collect_host_hardware(runner)
    assert hw.chip_brand  # falls back to platform.processor()
    assert hw.host_id_hash == ""
    os_info = telemetry.collect_host_os(runner)
    assert os_info.product_name == "macOS"
    assert os_info.build_version == ""
    repo = telemetry.collect_repo_state(".", runner)
    assert repo.git_sha is None
    assert repo.dirty is False


def test_runner_timeout_does_not_propagate():
    def hostile(argv: list[str]):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=5)

    assert telemetry.collect_thermal_snapshot(hostile).source == "unavailable"
    assert telemetry.host_id_hash(hostile) == ""


# ---------------------------------------------------------------------------
# power sampler meta
# ---------------------------------------------------------------------------


def test_collect_power_sampler_meta_includes_plist_sha(tmp_path):
    plist = tmp_path / "p.plist"
    plist.write_bytes(b"<plist></plist>")
    # ``powermetrics`` has no ``--version``; we fingerprint via ``codesign``
    # plus the binary's mtime + size, so the runner needs to answer codesign.
    runner = make_runner(
        {
            ("codesign", "-dv", "/usr/bin/powermetrics"): FakeProc(
                0,
                "",
                stderr=(
                    "Executable=/usr/bin/powermetrics\n"
                    "Signed Time=11 Oct 2025 at 09:27:31\n"
                ),
            )
        }
    )
    meta = telemetry.collect_power_sampler_meta(
        interval_ms=100,
        samplers=["cpu_power", "gpu_power", "ane_power"],
        baseline_seconds=1.0,
        plist_path=plist,
        sudo_cached=True,
        runner=runner,
    )
    assert meta.powermetrics_version and "signed=11 Oct 2025" in meta.powermetrics_version
    assert "powermetrics" not in meta.powermetrics_version.lower()  # no Usage banner regression
    assert meta.interval_ms == 100
    assert meta.sudo_cached is True
    assert meta.plist_sha256 and len(meta.plist_sha256) == 64
    assert meta.plist_path == str(plist)


# ---------------------------------------------------------------------------
# run conditions
# ---------------------------------------------------------------------------


def test_run_conditions_carry_start_and_end():
    runner = make_runner(
        {
            ("pmset", "-g", "batt"): FakeProc(0, "Now drawing from 'AC Power'\n-InternalBattery 80%"),
            ("pmset", "-g", "custom"): FakeProc(0, "displaysleep 10\nSleepDisabled 1\nlowpowermode 0"),
            ("pgrep", "-x", "caffeinate"): FakeProc(1, ""),
            ("pmset", "-g", "therm"): FakeProc(0, "CPU_Speed_Limit = 100"),
            ("ps", "-Ao", "pid,pcpu,comm", "-r"): FakeProc(0, "PID  %CPU COMMAND\n1 1.0 a\n"),
            ("sysctl", "-n", "kern.boottime"): FakeProc(0, "{ sec = 1700000000, usec = 0 }"),
        }
    )
    start = telemetry.snapshot_run_conditions_start("background quiet", runner)
    assert start.power_at_start is not None and start.power_at_start.ac_powered is True
    assert start.finished_at_iso is None
    end = telemetry.snapshot_run_conditions_end(start, runner)
    assert end.finished_at_iso is not None
    assert end.power_at_end is not None
    assert end.wall_duration_s is not None and end.wall_duration_s >= 0
