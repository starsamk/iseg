"""Label conversion utilities for iSeg-2017 annotations."""

from typing import Dict

import numpy as np

RAW_TO_INTERNAL_LABELS: Dict[int, int] = {
    0: 0,
    10: 1,
    150: 2,
    250: 3,
}


def remap_labels(labels: np.ndarray) -> np.ndarray:
    """Convert raw iSeg-2017 labels to contiguous internal class indices.

    The returned array has the same shape as ``labels`` and does not modify
    the input array.
    """

    raw_values = np.unique(labels)
    unknown_values = [
        value.item()
        for value in raw_values
        if value.item() not in RAW_TO_INTERNAL_LABELS
    ]
    if unknown_values:
        expected_values = sorted(RAW_TO_INTERNAL_LABELS)
        raise ValueError(
            f"Unknown label values: {unknown_values}. Expected: {expected_values}"
        )

    remapped = np.empty(labels.shape, dtype=np.uint8)
    for raw_value, internal_value in RAW_TO_INTERNAL_LABELS.items():
        remapped[labels == raw_value] = internal_value

    return remapped
