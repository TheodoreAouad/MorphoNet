"""Layer implementing affine function."""

from typing import Any
import torch
from torch import nn


class ScaleBias(nn.Module):
    """Simple affine layer."""

    weight: nn.Parameter
    bias: nn.Parameter

    def __init__(self, num_features: int, **kwargs: Any) -> None:
        """Initialize affine ScaleBias Layer."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features, 1, 1, **kwargs))
        self.bias = nn.Parameter(torch.zeros(num_features, 1, 1, **kwargs))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward method of the layer."""
        return batch * self.weight + self.bias
