"""All the losses supported."""

from typing import Callable
import pytorch_ssim  # type: ignore
from torch import nn, Tensor


def make_ssim_loss() -> Callable:
    """Instanciate SSIM loss."""
    ssim_loss = pytorch_ssim.SSIM(window_size=3)

    def loss_func(img1: Tensor, img2: Tensor) -> float:
        return 1.0 - ssim_loss(img1, img2)

    return loss_func


def make_mse_loss() -> Callable:
    """Instanciate MSE loss."""
    return nn.MSELoss()


def make_crossentropy_loss() -> Callable:
    """Instanciate cross-entropy loss."""
    return nn.CrossEntropyLoss()


LOSSES = {
    "ssim": make_ssim_loss,
    "mse": make_mse_loss,
    "crossentropy": make_crossentropy_loss,
}
