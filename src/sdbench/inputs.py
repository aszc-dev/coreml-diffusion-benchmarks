import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Recorded into DeterminismInputs so a contributor can verify they ran the same
# RNG, not just the same seed (R11.11).
RNG_KIND = "numpy.default_rng(PCG64)"

# Today the text-embedding tensor is sampled from a normal distribution (R0.1
# scope is UNet-only — text encoder is out of the timed path). Stamped into
# every record so the choice is auditable; switching to a real CLIP-derived
# embedding would bump this value, not silently change numerics.
TEXT_EMBEDDING_SOURCE = "random_normal"

DEFAULT_BATCH_SIZE = 2
DEFAULT_TIMESTEP = 500


@dataclass(frozen=True)
class SharedInput:
    latent: np.ndarray
    timestep: int
    text_embedding: np.ndarray


def generate_shared_input(seed: int, resolution: int, batch_size: int = DEFAULT_BATCH_SIZE) -> SharedInput:
    latent_size = resolution // 8
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((batch_size, 4, latent_size, latent_size), dtype=np.float32)
    text_embedding = rng.standard_normal((batch_size, 77, 768), dtype=np.float32)
    return SharedInput(latent=latent, timestep=DEFAULT_TIMESTEP, text_embedding=text_embedding)


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


def digest_shared_input(shared: SharedInput) -> dict[str, str]:
    """SHA-256 of the materialized fp32 arrays — the contract every backend sees (R11.11).

    A contributor running a different shared input cannot pass off the result as
    comparable: the per-record hashes won't match.
    """
    return {
        "latent": hashlib.sha256(shared.latent.tobytes()).hexdigest(),
        "text_embedding": hashlib.sha256(shared.text_embedding.tobytes()).hexdigest(),
    }


def sha256_npz_file(path: str | Path) -> str | None:
    """SHA-256 of the .npz file on disk, or None if it's missing."""
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
