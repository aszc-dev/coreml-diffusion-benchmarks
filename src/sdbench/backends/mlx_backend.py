from sdbench.adapter import UnavailableBackendAdapter


def build_adapter():
    return UnavailableBackendAdapter(
        "mlx",
        "MLX SD 1.5 UNet implementation is gated until the canonical configuration passes equivalence",
    )
