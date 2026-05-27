from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EquivalenceResult:
    mse: float
    cosine: float
    passed: bool


def compare_to_reference(
    output: np.ndarray,
    reference: np.ndarray,
    mse_max: float,
    cosine_min: float,
) -> EquivalenceResult:
    output_fp32 = np.asarray(output, dtype=np.float32).ravel()
    reference_fp32 = np.asarray(reference, dtype=np.float32).ravel()
    diff = output_fp32 - reference_fp32
    mse = float(np.mean(diff * diff))
    denom = float(np.linalg.norm(output_fp32) * np.linalg.norm(reference_fp32))
    cosine = 0.0 if denom == 0.0 else float(np.dot(output_fp32, reference_fp32) / denom)
    return EquivalenceResult(mse=mse, cosine=cosine, passed=mse < mse_max and cosine > cosine_min)
