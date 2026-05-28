import re
import subprocess
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ThermalState:
    throttled: bool
    source: str
    detail: str


def check_thermal_state(probe: Callable[[], str] | None = None) -> ThermalState:
    """Detect CPU thermal throttling via `pmset -g therm` (no sudo). (R5.6)

    Returns throttled=True only when a CPU speed limit below 100% is reported. If
    the probe is unavailable the state is reported as unknown (source
    "unavailable") rather than silently "fine", so a run is never trusted on an
    unverified thermal state. The probe is injectable for testing.
    """
    try:
        text = probe() if probe is not None else _pmset_therm()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return ThermalState(throttled=False, source="unavailable", detail="pmset unavailable; thermal state unknown")
    if not text:
        return ThermalState(throttled=False, source="unavailable", detail="pmset produced no output")

    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text)
    if match:
        limit = int(match.group(1))
        if limit < 100:
            return ThermalState(throttled=True, source="pmset", detail=f"CPU_Speed_Limit={limit}% (throttled)")
        return ThermalState(throttled=False, source="pmset", detail=f"CPU_Speed_Limit={limit}%")
    return ThermalState(throttled=False, source="pmset", detail="no thermal pressure reported")


def _pmset_therm() -> str:
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5)
    return out.stdout
