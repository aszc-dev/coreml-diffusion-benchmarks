import numpy as np

from sdbench.inputs import generate_shared_input, load_shared_input, save_shared_input


def test_generates_backend_neutral_shared_input_with_expected_sd15_shapes():
    shared = generate_shared_input(seed=7, resolution=512)

    assert shared.latent.shape == (2, 4, 64, 64)
    assert shared.timestep == 500
    assert shared.text_embedding.shape == (2, 77, 768)
    assert shared.latent.dtype == np.float32
    assert shared.text_embedding.dtype == np.float32


def test_shared_input_is_deterministic_and_round_trips(tmp_path):
    first = generate_shared_input(seed=7, resolution=512)
    second = generate_shared_input(seed=7, resolution=512)
    path = tmp_path / "shared_input.npz"

    save_shared_input(first, path)
    loaded = load_shared_input(path)

    np.testing.assert_array_equal(first.latent, second.latent)
    np.testing.assert_array_equal(first.text_embedding, loaded.text_embedding)
    assert loaded.timestep == first.timestep
