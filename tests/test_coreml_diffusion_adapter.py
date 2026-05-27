import numpy as np

from sdbench.backends.coreml_diffusion import CoreMLDiffusionAdapter
from sdbench.config import CellConfig


class FakeMLModel:
    last_inputs = None

    def __init__(self, path, compute_units=None):
        self.path = path
        self.compute_units = compute_units

    def predict(self, inputs):
        self.__class__.last_inputs = inputs
        return {"noise_pred": inputs["sample"].astype(np.float32) + 2.0}


class FakeCoreMLDiffusionAdapter(CoreMLDiffusionAdapter):
    @staticmethod
    def _coreml_compute_unit(compute_unit: str):
        return f"CU:{compute_unit}"


def test_coreml_diffusion_adapter_maps_native_text_embedding_layout(tmp_path):
    artifact = tmp_path / "original_gpu" / "local_sd15-ORIGINAL-fp16.mlmodelc"
    artifact.mkdir(parents=True)
    checkpoint = tmp_path / "sd15.safetensors"
    from safetensors.torch import save_file
    import torch

    save_file({"model.diffusion_model.weight": torch.zeros((2, 4), dtype=torch.float16)}, checkpoint)
    (artifact / "weights").mkdir()
    (artifact / "weights" / "weight.bin").write_bytes(b"0" * 16)
    adapter = FakeCoreMLDiffusionAdapter(checkpoint, artifact_root=tmp_path, compiled_model_cls=FakeMLModel)
    cfg = CellConfig(
        id="ours-gpu-fp16",
        backend="coreml_diffusion",
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
    assert FakeMLModel.last_inputs["sample"].shape == (2, 4, 64, 64)
    assert FakeMLModel.last_inputs["timestep"].shape == (2,)
    assert FakeMLModel.last_inputs["encoder_hidden_states"].shape == (2, 77, 768)
    np.testing.assert_array_equal(output, np.full((2, 4, 64, 64), 2.0, dtype=np.float32))
    assert adapter.model_size().effective_bits_per_parameter == 16.0
