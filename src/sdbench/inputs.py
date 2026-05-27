from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SharedInput:
    latent: np.ndarray
    timestep: int
    text_embedding: np.ndarray


def generate_shared_input(seed: int, resolution: int, batch_size: int = 2) -> SharedInput:
    latent_size = resolution // 8
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((batch_size, 4, latent_size, latent_size), dtype=np.float32)
    text_embedding = rng.standard_normal((batch_size, 77, 768), dtype=np.float32)
    return SharedInput(latent=latent, timestep=500, text_embedding=text_embedding)


def save_shared_input(shared: SharedInput, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        latent=shared.latent.astype(np.float32, copy=False),
        timestep=np.array(shared.timestep, dtype=np.int64),
        text_embedding=shared.text_embedding.astype(np.float32, copy=False),
    )


def load_shared_input(path: str | Path) -> SharedInput:
    data = np.load(Path(path))
    return SharedInput(
        latent=data["latent"].astype(np.float32, copy=False),
        timestep=int(data["timestep"]),
        text_embedding=data["text_embedding"].astype(np.float32, copy=False),
    )
