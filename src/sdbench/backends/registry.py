from pathlib import Path

from sdbench.adapter import UnavailableBackendAdapter
from sdbench.backends.apple_coreml import AppleCoreMLAdapter
from sdbench.backends.coreml_diffusion import CoreMLDiffusionAdapter
from sdbench.backends.diffusers_mps import DiffusersMpsAdapter


def build_default_adapters(
    artifact_root: str | Path = "artifacts",
    checkpoint_path: str | Path | None = None,
):
    root = Path(artifact_root)
    apple_adapter = (
        AppleCoreMLAdapter(Path(checkpoint_path), artifact_root=root / "apple_coreml")
        if checkpoint_path is not None
        else UnavailableBackendAdapter(
            "apple_coreml",
            "provide checkpoint_path when constructing the adapter registry",
            root / "apple_coreml",
        )
    )
    diffusers_adapter = (
        DiffusersMpsAdapter(Path(checkpoint_path))
        if checkpoint_path is not None
        else UnavailableBackendAdapter(
            "diffusers_mps",
            "provide checkpoint_path when constructing the adapter registry",
            root / "diffusers_mps",
        )
    )
    coreml_diffusion_adapter = (
        CoreMLDiffusionAdapter(Path(checkpoint_path), artifact_root=root / "coreml_diffusion")
        if checkpoint_path is not None
        else UnavailableBackendAdapter(
            "coreml_diffusion",
            "provide checkpoint_path when constructing the adapter registry",
            root / "coreml_diffusion",
        )
    )
    return {
        "apple_coreml": apple_adapter,
        "coreml_diffusion": coreml_diffusion_adapter,
        "diffusers_mps": diffusers_adapter,
        "mlx": UnavailableBackendAdapter(
            "mlx",
            "SD 1.5 MLX UNet support is intentionally implemented last after numerical cross-checks",
            root / "mlx",
        ),
    }
