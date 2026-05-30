from pathlib import Path

import numpy as np

from sdbench.adapter import RealizedConfig
from sdbench.sizing import safetensors_weight_size

_DTYPES = {"fp16": "float16", "fp32": "float32"}


class MlxAdapter:
    """SD 1.5 UNet on MLX (R3.4). Weights are loaded once via diffusers'
    LDM->diffusers conversion, remapped to the MLX tree (mlx_unet.load_weights),
    and the first graph build is forced in prepare() (R3.4.3). step() forces
    mx.eval before returning so no lazy work leaks into the timed window (R3.4.2).

    Heavy dependencies (mlx, diffusers, torch) are injected for testing and lazily
    imported otherwise, keeping import-time side effects out of the harness env."""

    name = "mlx"

    def __init__(
        self,
        checkpoint_path,
        mx_module=None,
        unet_module=None,
        state_dict_loader=None,
        compile=True,
        reference_unet_loader=None,
    ):
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self._mx = mx_module
        self._unet = unet_module
        self._state_dict_loader = state_dict_loader
        self._reference_unet_loader = reference_unet_loader
        self._compile = compile
        self._weights = None
        self._config = None
        self._dtype = None
        self._forward = None
        self._reference_unet = None

    def prepare(self, cfg) -> RealizedConfig:
        if cfg.compute_unit != "GPU":
            raise ValueError(f"mlx only supports compute_unit=GPU, got {cfg.compute_unit}")
        if cfg.attention != "NATIVE":
            raise ValueError(f"mlx only supports attention=NATIVE, got {cfg.attention}")
        if cfg.precision not in _DTYPES:
            raise ValueError(f"mlx supports precision {sorted(_DTYPES)}, got {cfg.precision}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")

        mx = self._load_mx()
        unet = self._load_unet()
        self._dtype = getattr(mx, _DTYPES[cfg.precision])
        self._config = unet.UNetConfig()
        weights = unet.load_weights(self._load_state_dict(), self._dtype)
        self._weights = weights
        config = self._config

        # The forward closes over the (constant) weights; timestep is an array input
        # so the compiled graph is reused across timesteps instead of re-traced.
        def forward(sample, context, timestep):
            return unet.unet_forward(weights, config, sample, timestep, context)

        self._forward = mx.compile(forward) if self._compile else forward

        # Force weight load, the first graph build, and the compile here, never in
        # step() (R3.4.3). Warm with the same shapes/dtypes the timed steps use.
        latent_size = cfg.resolution // 8
        warm_latent = mx.zeros((2, config.in_channels, latent_size, latent_size), dtype=self._dtype)
        warm_context = mx.zeros((2, 77, config.cross_attention_dim), dtype=self._dtype)
        warm = self._forward(warm_latent, warm_context, mx.array(1.0))
        mx.eval(warm)

        return RealizedConfig(
            compute_unit="GPU",
            attention="NATIVE",
            precision=cfg.precision,
            artifact_paths=[str(self.checkpoint_path)],
        )

    def step(self, latent: np.ndarray, timestep: int, text_embedding: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Adapter must be prepared before step()")
        mx = self._load_mx()
        sample = mx.array(np.asarray(latent, dtype=np.float32)).astype(self._dtype)
        context = mx.array(np.asarray(text_embedding, dtype=np.float32)).astype(self._dtype)
        output = self._forward(sample, context, mx.array(float(timestep)))
        mx.eval(output)  # force materialization before returning (R3.4.2)
        return np.asarray(output, dtype=np.float32)

    def reference_step(self, latent: np.ndarray, timestep: int, text_embedding: np.ndarray) -> np.ndarray:
        """fp32 CPU reference for equivalence comparison.

        Mirrors the other backends: build the diffusers UNet once on CPU in
        fp32 and run a single forward. Without this method the orchestrator
        falls back to ``shared_input.latent`` as the reference, which would
        compare the model output to its own noise input — meaningless."""
        unet = self._load_reference_unet()
        torch = self._load_torch_for_reference()
        latent_tensor = torch.from_numpy(np.asarray(latent, dtype=np.float32))
        text_tensor = torch.from_numpy(np.asarray(text_embedding, dtype=np.float32))
        with torch.no_grad():
            output = unet(latent_tensor, int(timestep), encoder_hidden_states=text_tensor).sample
        return output.float().cpu().numpy()

    def teardown(self) -> None:
        self._weights = None
        self._config = None
        self._forward = None
        self._reference_unet = None
        mx = self._mx
        if mx is not None and hasattr(mx, "clear_cache"):
            mx.clear_cache()

    def model_size(self):
        return safetensors_weight_size(
            self.checkpoint_path,
            key_prefixes=("model.diffusion_model.",),
            compute_precision="fp16",
        )

    def _load_mx(self):
        if self._mx is None:
            import mlx.core as mx

            self._mx = mx
        return self._mx

    def _load_unet(self):
        if self._unet is None:
            from sdbench.backends import mlx_unet

            self._unet = mlx_unet
        return self._unet

    def _load_state_dict(self):
        if self._state_dict_loader is not None:
            return self._state_dict_loader()
        import torch
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_single_file(
            str(self.checkpoint_path),
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        return unet.state_dict()

    def _load_reference_unet(self):
        if self._reference_unet is None:
            if self._reference_unet_loader is not None:
                self._reference_unet = self._reference_unet_loader()
            else:
                import torch
                from diffusers import UNet2DConditionModel

                unet = UNet2DConditionModel.from_single_file(
                    str(self.checkpoint_path),
                    torch_dtype=torch.float32,
                    local_files_only=True,
                )
                self._reference_unet = unet.to("cpu").eval()
        return self._reference_unet

    def _load_torch_for_reference(self):
        # ``torch`` is a hard dep of the harness env; importing here keeps the
        # MLX adapter import-clean for callers that only want ``step()``.
        import torch

        return torch


def build_adapter(checkpoint_path):
    return MlxAdapter(checkpoint_path)
