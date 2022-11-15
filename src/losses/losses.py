"""Supported losses."""

from typing import Callable
import pytorch_ssim  # type: ignore
import torch

from .base import Loss

class SSIM(Loss):
    """SSIM loss."""
    def __call__(self) -> Callable:
        ssim_loss = pytorch_ssim.SSIM(window_size=3)

        def loss_func(img1: torch.Tensor, img2: torch.Tensor) -> float:
            return 1.0 - ssim_loss(img1, img2)

        return loss_func

class MSE(Loss):
    """MSE loss."""
    def __call__(self) -> Callable:
        return torch.nn.MSELoss()

class CrossEntropy(Loss):
    """Cross-entropy loss."""
    def __call__(self) -> Callable:
        return torch.nn.CrossEntropyLoss()
