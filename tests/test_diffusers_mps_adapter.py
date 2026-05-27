from types import SimpleNamespace

import numpy as np
import pytest

from sdbench.backends.diffusers_mps import DiffusersMpsAdapter
from sdbench.config import CellConfig


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def to(self, device=None, dtype=None):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    float16 = "float16"
    float32 = "float32"

    def __init__(self):
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
        self.mps = SimpleNamespace(synchronize=self._synchronize, empty_cache=lambda: None)
        self.synchronized = False

    def from_numpy(self, array):
        return FakeTensor(array)

    def no_grad(self):
        return FakeNoGrad()

    def _synchronize(self):
        self.synchronized = True


class FakeUnet:
    def __init__(self):
        self.device = None
        self.evaluated = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, latent, timestep, encoder_hidden_states):
        return SimpleNamespace(sample=FakeTensor(latent.array + 1.0))


class FakeModelCls:
    loaded_path = None
    loaded_dtype = None
    loaded_dtypes = []
    instance = FakeUnet()

    @classmethod
    def from_single_file(cls, path, torch_dtype, local_files_only):
        cls.loaded_path = path
        cls.loaded_dtype = torch_dtype
        cls.loaded_dtypes.append(torch_dtype)
        assert local_files_only is True
        return FakeUnet()


def test_diffusers_mps_adapter_loads_unet_only_and_materializes_output(tmp_path):
    checkpoint = tmp_path / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")
    fake_torch = FakeTorch()
    adapter = DiffusersMpsAdapter(checkpoint, torch_module=fake_torch, model_cls=FakeModelCls)
    cfg = CellConfig(
        id="diffusers-mps-fp16",
        backend="diffusers_mps",
        compute_unit="MPS",
        attention="NATIVE",
        precision="fp16",
        resolution=512,
    )

    realized = adapter.prepare(cfg)
    output = adapter.step(
        np.zeros((1, 4, 64, 64), dtype=np.float32),
        500,
        np.zeros((1, 77, 768), dtype=np.float32),
    )

    assert realized.compute_unit == "MPS"
    assert realized.artifact_paths == [str(checkpoint)]
    assert FakeModelCls.loaded_path == str(checkpoint)
    assert "float16" in FakeModelCls.loaded_dtypes
    assert fake_torch.synchronized is True
    np.testing.assert_array_equal(output, np.ones((1, 4, 64, 64), dtype=np.float32))


def test_diffusers_mps_adapter_computes_cpu_fp32_reference(tmp_path):
    checkpoint = tmp_path / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")
    fake_torch = FakeTorch()
    FakeModelCls.loaded_dtypes = []
    adapter = DiffusersMpsAdapter(checkpoint, torch_module=fake_torch, model_cls=FakeModelCls)

    output = adapter.reference_step(
        np.zeros((1, 4, 64, 64), dtype=np.float32),
        500,
        np.zeros((1, 77, 768), dtype=np.float32),
    )

    assert "float32" in FakeModelCls.loaded_dtypes
    np.testing.assert_array_equal(output, np.ones((1, 4, 64, 64), dtype=np.float32))


def test_diffusers_mps_adapter_rejects_wrong_compute_unit(tmp_path):
    checkpoint = tmp_path / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")
    adapter = DiffusersMpsAdapter(checkpoint, torch_module=FakeTorch(), model_cls=FakeModelCls)
    cfg = CellConfig(
        id="bad",
        backend="diffusers_mps",
        compute_unit="CPU_AND_GPU",
        attention="NATIVE",
        precision="fp16",
        resolution=512,
    )

    with pytest.raises(ValueError, match="compute_unit=MPS"):
        adapter.prepare(cfg)
