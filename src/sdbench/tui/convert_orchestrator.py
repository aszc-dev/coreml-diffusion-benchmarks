"""Drive the two isolated conversion toolchains and cache their output.

coremltools 8 and 9 cannot share an interpreter, so each conversion runs in its
own pinned uv project (`envs/apple-ct8`, `envs/team-ct9`) via `uv run --project`.
The toolchain-version contrast (ct8 vs ct9) is the primary purpose of the study,
so conversion timing is captured per build (R9). Builds are cached: an artifact
is rebuilt only when missing or when the checkpoint it was built from changed
(tracked by a `.source.json` sidecar carrying the checkpoint SHA).

Command construction and the cache decision are pure and tested; the heavy
`uv run` is injected so tests never invoke a real toolchain.
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from sdbench.backends.apple_coreml import AppleCoreMLAdapter
from sdbench.backends.coreml_diffusion import CoreMLDiffusionAdapter

_DUMMY = "unused-for-path-resolution"


@dataclass(frozen=True)
class ConversionBuild:
    backend: str
    env_project: Path
    driver: Path
    output_dir: Path
    expected_artifact: Path
    timings_out: Path
    driver_args: list[str]
    cell_ids: list[str] = field(default_factory=list)

    @property
    def sidecar(self) -> Path:
        return self.output_dir / ".source.json"


def build_command(build: ConversionBuild, checkpoint: str | Path) -> list[str]:
    return [
        "uv", "run", "--project", str(build.env_project),
        "python", str(build.driver),
        "--checkpoint", str(checkpoint),
        "--output-dir", str(build.output_dir),
        "--timings-out", str(build.timings_out),
        *build.driver_args,
    ]


def plan_conversions(ws, cfg) -> list[ConversionBuild]:
    """Derive the unique CoreML builds the matrix needs, one per distinct artifact."""
    apple = AppleCoreMLAdapter(_DUMMY, artifact_root=ws.artifacts_dir / "apple_coreml")
    team = CoreMLDiffusionAdapter(_DUMMY, artifact_root=ws.artifacts_dir / "coreml_diffusion")
    by_artifact: dict[Path, ConversionBuild] = {}

    for cell in cfg.cells:
        try:
            if cell.backend == "apple_coreml":
                artifact = apple._artifact_path(cell)
                build = ConversionBuild(
                    backend="apple_coreml",
                    env_project=ws.root / "envs" / "apple-ct8",
                    driver=ws.root / "scripts" / "convert" / "apple_ct8.py",
                    output_dir=artifact.parent,
                    expected_artifact=artifact,
                    timings_out=artifact.parent / "conversion-timings.json",
                    driver_args=[
                        "--attention", cell.attention,
                        "--compute-unit", cell.compute_unit,
                        "--resolution", str(cell.resolution),
                    ],
                )
            elif cell.backend == "coreml_diffusion":
                artifact = team._artifact_path(cell)
                build = ConversionBuild(
                    backend="coreml_diffusion",
                    env_project=ws.root / "envs" / "team-ct9",
                    driver=ws.root / "scripts" / "convert" / "team_ct9.py",
                    output_dir=artifact.parent,
                    expected_artifact=artifact,
                    timings_out=artifact.parent / "conversion-timings.json",
                    driver_args=[
                        "--attention", cell.attention,
                        "--compute-unit", cell.compute_unit,
                        "--precision", cell.precision,
                        "--resolution", str(cell.resolution),
                    ],
                )
            else:
                continue
        except ValueError:
            # Cell has no buildable CoreML artifact path (e.g. a gated/unsupported
            # precision such as w8a8); it is N/A for conversion, not an error (R8.4).
            continue

        existing = by_artifact.get(build.expected_artifact)
        if existing is None:
            by_artifact[build.expected_artifact] = ConversionBuild(**{**build.__dict__, "cell_ids": [cell.id]})
        else:
            existing.cell_ids.append(cell.id)
    return list(by_artifact.values())


def is_cached(build: ConversionBuild, checkpoint_sha: str | None) -> bool:
    """True when the artifact exists and was built from the same checkpoint."""
    if not build.expected_artifact.exists():
        return False
    if checkpoint_sha is None:
        return True  # cannot verify source; trust an existing artifact
    if not build.sidecar.exists():
        return False
    try:
        recorded = json.loads(build.sidecar.read_text(encoding="utf-8")).get("checkpoint_sha256")
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == checkpoint_sha


def write_sidecar(build: ConversionBuild, checkpoint_sha: str | None) -> None:
    build.output_dir.mkdir(parents=True, exist_ok=True)
    build.sidecar.write_text(
        json.dumps({"checkpoint_sha256": checkpoint_sha, "built_s": time.time()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stream_run(cmd: list[str], on_line) -> None:
    """Run a command, streaming merged stdout+stderr line-by-line to `on_line`.

    Keeps the toolchain's output inside the UI (a Rich panel) instead of letting
    it spill onto the terminal.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        on_line(line.rstrip("\n"))
    code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def run_build(build: ConversionBuild, checkpoint: str | Path, checkpoint_sha: str | None, runner=subprocess.run, on_line=None) -> None:
    cmd = build_command(build, checkpoint)
    if on_line is not None:
        _stream_run(cmd, on_line)
    else:
        runner(cmd, check=True)
    write_sidecar(build, checkpoint_sha)


def convert_all(
    ws,
    cfg,
    checkpoint: str | Path,
    checkpoint_sha: str | None,
    *,
    force: bool = False,
    runner=subprocess.run,
    on_build=None,
    on_line=None,
    on_skip=None,
    on_done=None,
) -> list[ConversionBuild]:
    """Convert every needed build that is missing or stale. Returns the builds that ran.

    Optional callbacks drive a UI: ``on_skip(build)`` for cached builds,
    ``on_build(build, index, total)`` when a build starts, ``on_line(text)`` for
    each line of toolchain output (streamed when provided), ``on_done(build)``
    after a build converts successfully.
    """
    # Materialise the conversion drivers + isolated env definitions into the
    # workspace tree if they're missing. A repo checkout already has them
    # (no-op); a fresh `uv tool install` workspace gets them written from the
    # wheel's packaged data. Without this the `uv run --project envs/...`
    # invocation below would fail with 'can't open file ...apple_ct8.py'.
    from sdbench.config import materialise_convert_tree

    materialise_convert_tree(ws.root)

    to_run: list[ConversionBuild] = []
    skipped: list[ConversionBuild] = []
    for build in plan_conversions(ws, cfg):
        (to_run if (force or not is_cached(build, checkpoint_sha)) else skipped).append(build)

    if on_skip is not None:
        for build in skipped:
            on_skip(build)

    ran: list[ConversionBuild] = []
    for index, build in enumerate(to_run):
        if on_build is not None:
            on_build(build, index, len(to_run))
        run_build(build, checkpoint, checkpoint_sha, runner=runner, on_line=on_line)
        ran.append(build)
        if on_done is not None:
            on_done(build)
    return ran


def load_timings(build: ConversionBuild) -> dict:
    path = _find_timings_file(build)
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _find_timings_file(build: ConversionBuild) -> Path | None:
    if build.timings_out.exists():
        return build.timings_out
    if build.output_dir.exists():
        for candidate in sorted(build.output_dir.glob("*conversion*.json")):
            return candidate
    return None


def conversion_timings_by_cell(ws, cfg) -> dict[str, dict]:
    """Map each CoreML cell id to its build's conversion timings (R9), if recorded.

    Reads the timings JSON emitted by the conversion driver (graph capture /
    convert / first-load compile). Cells without a converted build are omitted.
    """
    keys = ("graph_capture_s", "convert_s", "first_load_compile_s")
    out: dict[str, dict] = {}
    for build in plan_conversions(ws, cfg):
        data = load_timings(build)
        if not data:
            continue
        timings = {key: data.get(key) for key in keys}
        for cell_id in build.cell_ids:
            out[cell_id] = timings
    return out
