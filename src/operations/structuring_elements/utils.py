"""Utility function for structuring elements construction."""

from typing import Tuple
from skimage import transform
import numpy as np


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
