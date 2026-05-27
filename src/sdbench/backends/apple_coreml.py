from pathlib import Path

import numpy as np

from sdbench.adapter import RealizedConfig, UnavailableBackendAdapter
from sdbench.sizing import ModelSize, artifact_size_bytes, effective_bits_per_parameter, safetensors_parameter_count


class AppleCoreMLAdapter:
    name = "apple_coreml"

    def __init__(self, checkpoint_path: str | Path, artifact_root: str | Path = "artifacts/apple_coreml", compiled_model_cls=None):
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.artifact_root = Path(artifact_root)
        self._compiled_model_cls = compiled_model_cls
        self._model = None
        self._reference_unet = None
        self._artifact_path_in_use = None

    def prepare(self, cfg) -> RealizedConfig:
        artifact_path = self._artifact_path(cfg)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Compiled CoreML artifact does not exist: {artifact_path}")
        compiled_model_cls = self._load_compiled_model_cls()
        self._model = compiled_model_cls(str(artifact_path), compute_units=self._coreml_compute_unit(cfg.compute_unit))
        self._artifact_path_in_use = artifact_path
        return RealizedConfig(
            compute_unit=cfg.compute_unit,
            attention=cfg.attention,
            precision=cfg.precision,
            artifact_paths=[str(artifact_path)],
        )

    def step(self, latent: np.ndarray, timestep: int, text_embedding: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Adapter must be prepared before step()")
        inputs = {
            "sample": np.asarray(latent, dtype=np.float16),
            "timestep": np.full((latent.shape[0],), timestep, dtype=np.float16),
            "encoder_hidden_states": np.asarray(text_embedding, dtype=np.float16).transpose(0, 2, 1)[:, :, None, :],
        }
        output = self._model.predict(inputs)
        if "noise_pred" in output:
            return np.asarray(output["noise_pred"], dtype=np.float32)
        return np.asarray(next(iter(output.values())), dtype=np.float32)

    def reference_step(self, latent: np.ndarray, timestep: int, text_embedding: np.ndarray) -> np.ndarray:
        import torch
        from diffusers import UNet2DConditionModel

        if self._reference_unet is None:
            self._reference_unet = UNet2DConditionModel.from_single_file(
                str(self.checkpoint_path),
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            self._reference_unet.to("cpu").eval()
        latent_tensor = torch.from_numpy(np.asarray(latent, dtype=np.float32))
        text_tensor = torch.from_numpy(np.asarray(text_embedding, dtype=np.float32))
        with torch.no_grad():
            return self._reference_unet(latent_tensor, int(timestep), encoder_hidden_states=text_tensor).sample.float().cpu().numpy()

    def teardown(self) -> None:
        self._model = None
        self._reference_unet = None
        self._artifact_path_in_use = None

    def model_size(self):
        if self._artifact_path_in_use is None:
            return None
        weight_path = self._artifact_path_in_use / "weights" / "weight.bin"
        weight_bytes = artifact_size_bytes(weight_path) if weight_path.exists() else artifact_size_bytes(self._artifact_path_in_use)
        parameter_count = safetensors_parameter_count(self.checkpoint_path, ("model.diffusion_model.",))
        return ModelSize(
            on_disk_size_bytes=artifact_size_bytes(self._artifact_path_in_use),
            weight_only_size_bytes=weight_bytes,
            effective_bits_per_parameter=effective_bits_per_parameter(weight_bytes, parameter_count),
            compute_precision="fp16",
        )

    def _artifact_path(self, cfg) -> Path:
        if cfg.attention == "ORIGINAL" and cfg.compute_unit == "CPU_AND_GPU":
            return self.artifact_root / "original_gpu" / "Stable_Diffusion_version_local_sd15_unet.mlmodelc"
        if cfg.attention == "SPLIT_EINSUM_V2" and cfg.compute_unit == "CPU_AND_NE":
            return self.artifact_root / "split_einsum_v2_ane" / "Stable_Diffusion_version_local_sd15_unet.mlmodelc"
        raise ValueError(f"Unsupported Apple CoreML cell: attention={cfg.attention}, compute_unit={cfg.compute_unit}")

    def _load_compiled_model_cls(self):
        if self._compiled_model_cls is None:
            import coremltools as ct

            self._compiled_model_cls = ct.models.CompiledMLModel
        return self._compiled_model_cls

    @staticmethod
    def _coreml_compute_unit(compute_unit: str):
        import coremltools as ct

        return ct.ComputeUnit[compute_unit]


def build_adapter(checkpoint_path: str | Path | None = None, artifact_root: str | Path = "artifacts/apple_coreml"):
    if checkpoint_path is None:
        return UnavailableBackendAdapter(
            "apple_coreml",
            "provide checkpoint_path when constructing the adapter registry",
        )
    return AppleCoreMLAdapter(checkpoint_path, artifact_root=artifact_root)
