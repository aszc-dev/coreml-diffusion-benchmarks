from pathlib import Path

import numpy as np

from sdbench.adapter import RealizedConfig
from sdbench.sizing import safetensors_weight_size


class DiffusersMpsAdapter:
    name = "diffusers_mps"

    def __init__(self, checkpoint_path: str | Path, torch_module=None, model_cls=None):
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self._torch = torch_module
        self._model_cls = model_cls
        self._unet = None
        self._reference_unet = None

    def prepare(self, cfg) -> RealizedConfig:
        if cfg.compute_unit != "MPS":
            raise ValueError(f"diffusers_mps only supports compute_unit=MPS, got {cfg.compute_unit}")
        if cfg.attention != "NATIVE":
            raise ValueError(f"diffusers_mps only supports attention=NATIVE, got {cfg.attention}")
        if cfg.precision != "fp16":
            raise ValueError(f"diffusers_mps only supports precision=fp16, got {cfg.precision}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")

        torch = self._load_torch()
        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS backend is not available")

        model_cls = self._load_model_cls()
        self._unet = model_cls.from_single_file(
            str(self.checkpoint_path),
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self._unet.to("mps").eval()
        return RealizedConfig(
            compute_unit="MPS",
            attention="NATIVE",
            precision="fp16",
            artifact_paths=[str(self.checkpoint_path)],
        )

    def step(self, latent: np.ndarray, timestep: int, text_embedding: np.ndarray) -> np.ndarray:
        if self._unet is None:
            raise RuntimeError("Adapter must be prepared before step()")
        return self._run_unet(self._unet, "mps", self._load_torch().float16, latent, timestep, text_embedding)

    def reference_step(self, latent: np.ndarray, timestep: int, text_embedding: np.ndarray) -> np.ndarray:
        torch = self._load_torch()
        if self._reference_unet is None:
            model_cls = self._load_model_cls()
            self._reference_unet = model_cls.from_single_file(
                str(self.checkpoint_path),
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            self._reference_unet.to("cpu").eval()
        return self._run_unet(self._reference_unet, "cpu", torch.float32, latent, timestep, text_embedding)

    def _run_unet(
        self,
        unet,
        device: str,
        dtype,
        latent: np.ndarray,
        timestep: int,
        text_embedding: np.ndarray,
    ) -> np.ndarray:
        torch = self._load_torch()
        text_tensor = torch.from_numpy(np.asarray(text_embedding, dtype=np.float32)).to(device=device, dtype=dtype)
        latent_tensor = torch.from_numpy(np.asarray(latent, dtype=np.float32)).to(device=device, dtype=dtype)
        with torch.no_grad():
            output = unet(latent_tensor, int(timestep), encoder_hidden_states=text_tensor).sample
            if device == "mps":
                torch.mps.synchronize()
            return output.float().cpu().numpy()

    def teardown(self) -> None:
        self._unet = None
        self._reference_unet = None
        torch = self._torch
        if torch is not None and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def model_size(self):
        return safetensors_weight_size(
            self.checkpoint_path,
            key_prefixes=("model.diffusion_model.",),
            compute_precision="fp16",
        )

    def _load_torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _load_model_cls(self):
        if self._model_cls is None:
            from diffusers import UNet2DConditionModel

            self._model_cls = UNet2DConditionModel
        return self._model_cls


def build_adapter(checkpoint_path: str | Path):
    return DiffusersMpsAdapter(checkpoint_path)
