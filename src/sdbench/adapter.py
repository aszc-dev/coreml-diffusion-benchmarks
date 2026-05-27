from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RealizedConfig:
    compute_unit: str
    attention: str
    precision: str
    artifact_paths: list[str]


class BackendAdapter(Protocol):
    name: str

    def prepare(self, cfg) -> RealizedConfig:
        """Load weights, compile or warm the model, and return realized config."""
        ...

    def step(self, latent: np.ndarray, timestep: int, text_embedding: np.ndarray) -> np.ndarray:
        """Run exactly one UNet forward pass and return host-materialized output."""
        ...

    def teardown(self) -> None:
        """Release backend resources."""
        ...


class UnavailableBackendAdapter:
    def __init__(self, name: str, reason: str, artifact_root: Path | None = None):
        self.name = name
        self.reason = reason
        self.artifact_root = artifact_root

    def prepare(self, cfg):
        raise RuntimeError(f"{self.name} backend is unavailable: {self.reason}")

    def step(self, latent: np.ndarray, timestep: int, text_embedding: np.ndarray) -> np.ndarray:
        raise RuntimeError(f"{self.name} backend is unavailable")

    def teardown(self) -> None:
        pass
