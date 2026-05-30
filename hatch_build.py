"""Hatch build hook: stamp the current git SHA into ``sdbench._build_info``.

Runs at wheel/sdist build time and rewrites the placeholder so the installed
package can report a stable harness commit identifier through the manifest,
even when the contributor's workspace is not a git clone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    return out or None


class StampBuildSHAHook(BuildHookInterface):
    """Overwrite ``src/sdbench/_build_info.py`` with the current commit SHA.

    The hook restores the original file in ``finalize`` so the stamping is a
    pure side-effect of building a wheel, not a working-tree mutation."""

    PLUGIN_NAME = "stamp-build-sha"

    def initialize(self, version: str, build_data: dict) -> None:  # noqa: D401
        repo_root = Path(self.root)
        sha = _git(["rev-parse", "HEAD"], repo_root)
        describe = _git(["describe", "--always", "--dirty", "--tags"], repo_root)
        target = repo_root / "src" / "sdbench" / "_build_info.py"
        if not target.exists():
            return
        self._original = target.read_text(encoding="utf-8")
        target.write_text(
            self._original
            .replace(
                'BUILD_GIT_SHA: str | None = None',
                f'BUILD_GIT_SHA: str | None = {sha!r}',
            )
            .replace(
                'BUILD_GIT_DESCRIBE: str | None = None',
                f'BUILD_GIT_DESCRIBE: str | None = {describe!r}',
            ),
            encoding="utf-8",
        )

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:  # noqa: D401
        target = Path(self.root) / "src" / "sdbench" / "_build_info.py"
        original = getattr(self, "_original", None)
        if original is not None and target.exists():
            target.write_text(original, encoding="utf-8")
