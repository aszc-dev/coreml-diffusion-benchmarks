"""Back-compat shim around :mod:`sdbench.telemetry` for the pre-run throttle gate (R5.6).

The thermal snapshot is now a single source in :mod:`sdbench.telemetry`. This module
keeps the historic ``check_thermal_state`` / ``ThermalState`` surface so existing
callers (``run_cmd``, tests) keep working unchanged.
"""

from dataclasses import dataclass
from typing import Callable

from sdbench import telemetry


@dataclass(frozen=True)
class ThermalState:
    throttled: bool
    source: str
    detail: str


def check_thermal_state(probe: Callable[[], str] | None = None) -> ThermalState:
    """Detect CPU thermal throttling via ``pmset -g therm`` (no sudo, R5.6).

    Returns ``throttled=True`` only when a CPU speed limit below 100% is reported. If
    the probe is unavailable the state is reported as unknown (source
    ``"unavailable"``) rather than silently "fine", so a run is never trusted on an
    unverified thermal state. The probe is injectable for testing.
    """
    if probe is not None:
        # Legacy test/path: parse a caller-supplied pmset string directly.
        # Probe failures degrade to "unavailable" rather than aborting (R5.6).
        try:
            text = probe()
        except (FileNotFoundError, OSError):
            return ThermalState(throttled=False, source="unavailable", detail="pmset unavailable; thermal state unknown")
        snapshot = _snapshot_from_text(text)
    else:
        snapshot = telemetry.collect_thermal_snapshot()
    return ThermalState(throttled=snapshot.throttled, source=snapshot.source, detail=snapshot.detail)


def _snapshot_from_text(text: str) -> telemetry.ThermalSnapshot:
    import re

    if not text:
        return telemetry.ThermalSnapshot(None, False, "unavailable", "pmset produced no output")
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text)
    if not match:
        return telemetry.ThermalSnapshot(None, False, "pmset", "no thermal pressure reported")
    limit = int(match.group(1))
    throttled = limit < 100
    detail = f"CPU_Speed_Limit={limit}%" + (" (throttled)" if throttled else "")
    return telemetry.ThermalSnapshot(limit, throttled, "pmset", detail)
