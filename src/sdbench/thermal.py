from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalState:
    throttled: bool
    source: str
    detail: str


def check_thermal_state() -> ThermalState:
    return ThermalState(
        throttled=False,
        source="not-implemented",
        detail="Thermal throttling probe is platform-specific and must be wired before trusted measurements.",
    )
