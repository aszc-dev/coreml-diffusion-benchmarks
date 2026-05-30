"""Build-time metadata stamped into the wheel.

``BUILD_GIT_SHA`` is rewritten by ``hatch_build.py`` at wheel-build time so a
contributor's manifest can point reviewers at the exact harness commit that
produced their numbers, even when the tool is installed via ``uv tool install``
from a published wheel (where the workspace is not a git clone). When the
package is imported from a source checkout that has never been built into a
wheel, the placeholder ``None`` is honest about the absence of a stamp."""

BUILD_GIT_SHA: str | None = None
BUILD_GIT_DESCRIBE: str | None = None
