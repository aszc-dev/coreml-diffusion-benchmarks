from types import SimpleNamespace

import numpy as np

from sdbench.backends.apple_coreml import AppleCoreMLAdapter
from sdbench.config import CellConfig


class FakeMLModel:
    loaded_path = None
    loaded_compute_units = None
    last_inputs = None

    def __init__(self, path, compute_units=None):
        self.__class__.loaded_path = path
        self.__class__.loaded_compute_units = compute_units

    def predict(self, inputs):
        self.__class__.last_inputs = inputs
        return {"noise_pred": inputs["sample"].astype(np.float32) + 2.0}


class FakeAppleCoreMLAdapter(AppleCoreMLAdapter):
    @staticmethod
    def _coreml_compute_unit(compute_unit: str):
        return f"CU:{compute_unit}"


def test_apple_coreml_adapter_loads_compiled_artifact_and_maps_inputs(tmp_path):
    artifact = tmp_path / "original_gpu" / "Stable_Diffusion_version_local_sd15_unet.mlmodelc"
    artifact.mkdir(parents=True)
    checkpoint = tmp_path / "sd15.safetensors"
    from safetensors.torch import save_file
    import torch

    save_file({"model.diffusion_model.weight": torch.zeros((2, 4), dtype=torch.float16)}, checkpoint)
    (artifact / "weights").mkdir()
    (artifact / "weights" / "weight.bin").write_bytes(b"0" * 16)
    adapter = FakeAppleCoreMLAdapter(checkpoint, artifact_root=tmp_path, compiled_model_cls=FakeMLModel)
    cfg = CellConfig(
        id="apple-gpu-fp16",
        backend="apple_coreml",
        compute_unit="CPU_AND_GPU",
        attention="ORIGINAL",
        precision="fp16",
        resolution=512,
    )

    realized = adapter.prepare(cfg)
    output = adapter.step(
        np.zeros((2, 4, 64, 64), dtype=np.float32),
        500,
        np.zeros((2, 77, 768), dtype=np.float32),
    )

    assert realized.artifact_paths == [str(artifact)]
    assert FakeMLModel.loaded_path == str(artifact)
    assert FakeMLModel.loaded_compute_units == "CU:CPU_AND_GPU"
    assert FakeMLModel.last_inputs["sample"].shape == (2, 4, 64, 64)
    assert FakeMLModel.last_inputs["timestep"].shape == (2,)
    assert FakeMLModel.last_inputs["encoder_hidden_states"].shape == (2, 768, 1, 77)
    np.testing.assert_array_equal(output, np.full((2, 4, 64, 64), 2.0, dtype=np.float32))
    size = adapter.model_size()
    assert size.on_disk_size_bytes >= 16
    assert size.weight_only_size_bytes == 16
    assert size.effective_bits_per_parameter == 16.0
