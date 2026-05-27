from dataclasses import dataclass


@dataclass(frozen=True)
class ConversionTimings:
    graph_capture_s: float | None
    convert_s: float | None
    first_load_compile_s: float | None
