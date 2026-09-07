"""Intensity normalization utilities for MRI volumes."""

import numpy as np


def normalize_nonzero_intensity(volume: np.ndarray) -> np.ndarray:
    """Standardize non-zero voxels and preserve zero voxels.

    The mean and standard deviation are computed over the non-zero voxels of
    this volume only. The input array is not modified.
    """

    nonzero_mask = volume != 0
    if not np.any(nonzero_mask):
        raise ValueError("Cannot normalize a volume without non-zero voxels")

    nonzero_values = volume[nonzero_mask].astype(np.float32)
    mean = np.mean(nonzero_values, dtype=np.float64)
    standard_deviation = np.std(nonzero_values, dtype=np.float64)
    if standard_deviation == 0:
        raise ValueError(
            "Cannot normalize a volume with zero variance among non-zero voxels"
        )

    normalized = np.zeros(volume.shape, dtype=np.float32)
    normalized[nonzero_mask] = (nonzero_values - mean) / standard_deviation
    return normalized
