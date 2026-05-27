from types import SimpleNamespace

import numpy as np
import pytest

from sdbench.backends.mlx_backend import MlxAdapter
from sdbench.config import CellConfig


class FakeArray:
    """Stand-in for an mlx array: ignores dtype casts and materializes to numpy."""

    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def astype(self, _dtype):
        return self

    def __array__(self, dtype=None):
        return self.value if dtype is None else self.value.astype(dtype)


class FakeMx:
    float16 = "float16"
    float32 = "float32"

    def __init__(self):
        self.eval_calls = 0
        self.cleared = False

    def zeros(self, shape, dtype=None):
        return FakeArray(np.zeros(shape, dtype=np.float32))

    def array(self, value):
        return FakeArray(value)

    def eval(self, *_args):
        self.eval_calls += 1

    def clear_cache(self):
        self.cleared = True


class FakeUnetModule:
    """Stand-in for sdbench.backends.mlx_unet with the two functions the adapter uses."""

    def __init__(self):
        self.loaded_dtype = None
        self.forward_calls = []

    def UNetConfig(self):  # noqa: N802 - mirrors the real dataclass name
        return SimpleNamespace(in_channels=4, cross_attention_dim=768)

    def load_weights(self, state_dict, dtype):
        self.loaded_dtype = dtype
        return {"n": len(state_dict)}

    def unet_forward(self, weights, cfg, sample, timestep, context):
        # Echo the input latent so the adapter's NCHW passthrough is observable.
        self.forward_calls.append((timestep, np.asarray(sample).shape))
        return FakeArray(np.asarray(sample) + 1.0)


def _cfg(**overrides):
    base = dict(
        id="mlx-gpu-fp16",
        backend="mlx",
        compute_unit="GPU",
        attention="NATIVE",
        precision="fp16",
        resolution=512,
    )
    base.update(overrides)
    return CellConfig(**base)


def _adapter(tmp_path, mx=None, unet=None):
    checkpoint = tmp_path / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")
    return MlxAdapter(
        checkpoint,
        mx_module=mx or FakeMx(),
        unet_module=unet or FakeUnetModule(),
        state_dict_loader=lambda: {"a": 1, "b": 2},
    )


def test_prepare_loads_weights_and_forces_first_graph_build(tmp_path):
    mx, unet = FakeMx(), FakeUnetModule()
    adapter = _adapter(tmp_path, mx, unet)

    realized = adapter.prepare(_cfg())

    assert realized.compute_unit == "GPU"
    assert realized.attention == "NATIVE"
    assert realized.precision == "fp16"
    assert realized.artifact_paths == [str(adapter.checkpoint_path)]
    assert unet.loaded_dtype == "float16"  # fp16 cell maps to mx.float16
    assert unet.forward_calls and mx.eval_calls >= 1  # warmup forward + forced eval (R3.4.3)


def test_step_forces_eval_and_returns_host_nchw(tmp_path):
    mx, unet = FakeMx(), FakeUnetModule()
    adapter = _adapter(tmp_path, mx, unet)
    adapter.prepare(_cfg())
    evals_after_prepare = mx.eval_calls

    latent = np.zeros((2, 4, 64, 64), dtype=np.float32)
    output = adapter.step(latent, 500, np.zeros((2, 77, 768), dtype=np.float32))

    assert isinstance(output, np.ndarray) and output.dtype == np.float32
    assert output.shape == (2, 4, 64, 64)
    np.testing.assert_array_equal(output, latent + 1.0)
    assert mx.eval_calls == evals_after_prepare + 1  # step forces eval before returning (R3.4.2)
    assert unet.forward_calls[-1][0] == 500


def test_step_before_prepare_raises(tmp_path):
    adapter = _adapter(tmp_path)
    with pytest.raises(RuntimeError, match="prepared before step"):
        adapter.step(np.zeros((2, 4, 64, 64), dtype=np.float32), 500, np.zeros((2, 77, 768), dtype=np.float32))


def test_prepare_rejects_wrong_compute_unit(tmp_path):
    adapter = _adapter(tmp_path)
    with pytest.raises(ValueError, match="compute_unit=GPU"):
        adapter.prepare(_cfg(compute_unit="MPS"))


def test_prepare_rejects_unsupported_precision(tmp_path):
    adapter = _adapter(tmp_path)
    with pytest.raises(ValueError, match="precision"):
        adapter.prepare(_cfg(precision="w4"))


def test_prepare_missing_checkpoint_raises(tmp_path):
    adapter = MlxAdapter(
        tmp_path / "absent.safetensors",
        mx_module=FakeMx(),
        unet_module=FakeUnetModule(),
        state_dict_loader=lambda: {},
    )
    with pytest.raises(FileNotFoundError):
        adapter.prepare(_cfg())


def test_teardown_releases_weights_and_clears_cache(tmp_path):
    mx, unet = FakeMx(), FakeUnetModule()
    adapter = _adapter(tmp_path, mx, unet)
    adapter.prepare(_cfg())

    adapter.teardown()

    assert adapter._weights is None
    assert mx.cleared is True
