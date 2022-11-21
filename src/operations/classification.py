"""Module containing the code to prepare targets for classification."""

from typing import Tuple

import numpy as np
import torch

from .base import Operation


class Classification(Operation):
    """Class containing preprocessing for classification."""

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = np.arange(inputs.shape[0]) * 10 + targets.numpy()
        targets_np = np.zeros(inputs.shape[0] * 10)
        targets_np[idx] = 1

        return inputs, torch.from_numpy(targets_np.reshape(-1, 10))
