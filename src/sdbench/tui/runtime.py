"""Detect whether the heavy benchmark dependencies are installed.

The published tool installs light (rich/questionary/numpy/typer) so
`uv tool install` is fast; the heavy runtime stack (torch, coremltools, …) is an
optional extra pulled on first real use. This reports, in human terms, what is
missing and how to get it, instead of crashing with an ImportError deep inside an
adapter.
"""

from dataclasses import dataclass
from importlib.util import find_spec

RUNTIME_MODULES = ("torch", "coremltools", "diffusers", "transformers", "safetensors")

INSTALL_HINT = (
    "Heavy benchmark dependencies are not installed. Add them with:\n"
    "  uv tool install 'coreml-diffusion-benchmarks[bench]'   (published tool)\n"
    "  uv sync --extra bench                                   (from a repo checkout)"
)


@dataclass(frozen=True)
class RuntimeStatus:
    missing: list[str]

    @property
    def ready(self) -> bool:
        return not self.missing


def check_runtime() -> RuntimeStatus:
    return RuntimeStatus(missing=[name for name in RUNTIME_MODULES if find_spec(name) is None])


def ensure_runtime(console=None) -> bool:
    status = check_runtime()
    if status.ready:
        return True
    if console is not None:
        console.print(f"[sdbench.warn]Missing: {', '.join(status.missing)}.[/]\n{INSTALL_HINT}")
    return False
