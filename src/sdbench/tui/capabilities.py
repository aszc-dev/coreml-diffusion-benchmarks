"""Host capability detection for resolving matrix `requires` gates (R8.4).

A gated cell is recorded N/A — not failed — when the host lacks the capability
(e.g. W8A8 needs ANE activation quant on M4/A17-class silicon; a torch.compile
cell needs torch >= 2.4). This module decides, per host, whether a gate is met,
unmet, or undeterminable, so the config UI can lock the cell with a clear reason.
"""

import platform
import re
import subprocess
from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True)
class Capabilities:
    chip: str
    apple_generation: int | None  # M-series number (M2 -> 2), None if not Apple Silicon / unknown

    def supports(self, requirement: str) -> bool | None:
        """True/False if known, None if undeterminable on this host."""
        if requirement == "ane_activation_quant":
            if self.apple_generation is None:
                return None
            return self.apple_generation >= 4  # M4 / A17 Pro class (R3.2.4)
        if requirement == "torch_compile":
            return _torch_at_least(2, 4)
        return None


@dataclass(frozen=True)
class GateStatus:
    state: str  # "ok" | "unmet" | "unknown"
    detail: str

    @property
    def selectable(self) -> bool:
        return self.state == "ok"


def detect_capabilities(brand: str | None = None) -> Capabilities:
    chip = brand if brand is not None else _chip_brand()
    match = re.search(r"Apple M(\d+)", chip or "")
    return Capabilities(chip=chip or "unknown", apple_generation=int(match.group(1)) if match else None)


def evaluate_gate(caps: Capabilities, requires: dict | None) -> GateStatus:
    if not requires:
        return GateStatus(state="ok", detail="")
    unmet: list[str] = []
    unknown: list[str] = []
    for name, expected in requires.items():
        supported = caps.supports(name)
        if supported is None:
            unknown.append(name)
        elif bool(expected) and not supported:
            unmet.append(name)
    if unmet:
        return GateStatus(state="unmet", detail=f"needs {', '.join(unmet)} (unavailable on {caps.chip})")
    if unknown:
        return GateStatus(state="unknown", detail=f"cannot verify {', '.join(unknown)} on {caps.chip}")
    return GateStatus(state="ok", detail="")


def _chip_brand() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    return platform.processor() or platform.machine()


def _torch_at_least(major: int, minor: int) -> bool | None:
    try:
        version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        return None
    parts = version.split(".")
    try:
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (IndexError, ValueError):
        return None
