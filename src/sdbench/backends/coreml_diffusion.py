from pathlib import Path

import numpy as np

from sdbench.adapter import RealizedConfig, UnavailableBackendAdapter
from sdbench.sizing import ModelSize, artifact_size_bytes, effective_bits_per_parameter, safetensors_parameter_count


class CoreMLDiffusionAdapter:
    name = "coreml_diffusion"

    def __init__(self, checkpoint_path: str | Path, artifact_root: str | Path = "artifacts/coreml_diffusion", compiled_model_cls=None):
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.artifact_root = Path(artifact_root)
        self._compiled_model_cls = compiled_model_cls
        self._model = None
        self._reference_unet = None
        self._artifact_path_in_use = None
        self._precision_in_use = None

    def prepare(self, cfg) -> RealizedConfig:
        artifact_path = self._artifact_path(cfg)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Compiled CoreML artifact does not exist: {artifact_path}")
        compiled_model_cls = self._load_compiled_model_cls()
        self._model = compiled_model_cls(str(artifact_path), compute_units=self._coreml_compute_unit(cfg.compute_unit))
        self._artifact_path_in_use = artifact_path
        self._precision_in_use = cfg.precision
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
            "encoder_hidden_states": np.asarray(text_embedding, dtype=np.float16),
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
        self._precision_in_use = None

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
            compute_precision=self._precision_in_use or "fp16",
        )

    def _artifact_path(self, cfg) -> Path:
        if cfg.attention == "ORIGINAL" and cfg.compute_unit == "CPU_AND_GPU" and cfg.precision == "fp16":
            return self.artifact_root / "original_gpu" / "local_sd15-ORIGINAL-fp16.mlmodelc"
        if cfg.attention == "SPLIT_EINSUM_V2" and cfg.compute_unit == "CPU_AND_NE" and cfg.precision == "fp16":
            return self.artifact_root / "split_einsum_v2_ane" / "local_sd15-SPLIT_EINSUM_V2-fp16.mlmodelc"
        if cfg.attention == "SPLIT_EINSUM_V2" and cfg.compute_unit == "CPU_AND_NE" and cfg.precision == "w4":
            return self.artifact_root / "split_einsum_v2_ane_w4" / "local_sd15-SPLIT_EINSUM_V2-w4.mlmodelc"
        raise ValueError(
            f"Unsupported coreml-diffusion cell: attention={cfg.attention}, compute_unit={cfg.compute_unit}, precision={cfg.precision}"
        )

    def _load_compiled_model_cls(self):
        if self._compiled_model_cls is None:
            import coremltools as ct

            self._compiled_model_cls = ct.models.CompiledMLModel
        return self._compiled_model_cls

    @staticmethod
    def _coreml_compute_unit(compute_unit: str):
        import coremltools as ct

        return ct.ComputeUnit[compute_unit]


def build_adapter(checkpoint_path: str | Path | None = None, artifact_root: str | Path = "artifacts/coreml_diffusion"):
    if checkpoint_path is None:
        return UnavailableBackendAdapter(
            "coreml_diffusion",
            "provide checkpoint_path when constructing the adapter registry",
        )
    return CoreMLDiffusionAdapter(checkpoint_path, artifact_root=artifact_root)
