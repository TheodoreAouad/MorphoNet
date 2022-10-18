"""Utility function for structuring elements construction."""

from typing import Tuple
from skimage import transform
import numpy as np


def center_in(
    center: np.ndarray, filter_shape: Tuple[int, int], dtype: str = "float32"
) -> np.ndarray:
    """Pad the computed center form to match the desired filter shape."""
    pad_before = (filter_shape[0] - center.shape[0]) // 2, (
        filter_shape[1] - center.shape[1]
    ) // 2
    pad_after = filter_shape[0] - pad_before[0], filter_shape[1] - pad_before[1]

    centered = np.zeros(shape=filter_shape, dtype=dtype)
    centered[
        pad_before[0] : pad_after[0], pad_before[1] : pad_after[1]
    ] = center

    return centered


def shape_aa(
    structuring_element: np.ndarray, target_shape: Tuple[int, int]
) -> np.ndarray:
    """Resize the filter to the desired shape and smooth it if required."""
    return transform.resize(
        structuring_element,
        target_shape,
        preserve_range=True,
        anti_aliasing=True,
    )
