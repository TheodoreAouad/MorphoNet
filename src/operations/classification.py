"""Module containing the code to prepare targets for classification."""

import numpy as np

from .base import Operation

class Classification(Operation):
    """Class containing preprocessing for classification."""
    def __call__(self, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        idx = np.arange(inputs.shape[0]) * 10 + targets
        targets = np.zeros(inputs.shape[0] * 10)
        targets[idx] = 1
        return targets.reshape(-1, 10)