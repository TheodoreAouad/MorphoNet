"""Utility functions."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torchvision

PRECISIONS_TORCH = {"f32": torch.float32, "f64": torch.float64}
PRECISIONS_NP = {"f32": "float32", "f64": "float64"}


def fit_nchw(tensor: torch.Tensor, include_c: bool = False) -> torch.Tensor:
    """Change tensor to NCHV format."""
    return torchvision.transforms.Lambda(lambda x: _fit_nchw(x, include_c))(
        tensor
    )


def _fit_nchw(tensor: torch.Tensor, include_c: bool = False) -> torch.Tensor:
    """Change tensor to NCHV format (helper)."""
    if len(tensor.size()) == 4:
        return tensor  # Already in NCHW format

    if len(tensor.size()) == 2:  # Only H and W dimensions
        return tensor[None, None, :, :]

    if include_c:  # 3D data (C, H, W) with only one sample
        return tensor[None, :, :, :]

    return tensor[:, None, :, :]  # 2D data with N samples (N, C, W)


def rmse(
    array_x: np.ndarray, array_y: np.ndarray, adjust_range: bool = True
) -> float:
    """Calculate RMSE between two arrays."""
    if adjust_range:
        array_x = (array_x - np.min(array_x)) / (
            np.max(array_x) - np.min(array_x)
        )
        array_y = (array_y - np.min(array_y)) / (
            np.max(array_y) - np.min(array_y)
        )

    sum_ = np.sum(np.square(array_x - array_y))

    return np.sqrt(sum_ / np.prod(array_x.shape))


def snr(
    noised: np.ndarray, target: np.ndarray, adjust_range: bool = True
) -> float:
    """Calculate SNR (dB) between two arrays."""
    if adjust_range:
        noised = (noised - np.min(noised)) / (np.max(noised) - np.min(noised))
        target = (target - np.min(target)) / (np.max(target) - np.min(target))

    return 10 * np.log10(np.sum(noised**2) / np.sum((target - noised) ** 2))


def psnr(
    noised: np.ndarray,
    target: np.ndarray,
    adjust_range: bool = True,
    image_max: float = 1.0,
) -> float:
    """Calculate PSNR (dB) between two arrays."""
    if adjust_range:
        noised = (noised - np.min(noised)) / (np.max(noised) - np.min(noised))
        target = (target - np.min(target)) / (np.max(target) - np.min(target))

    mse = np.mean((target - noised) ** 2, dtype=np.float64)
    return 10 * np.log10((image_max ** 2) / mse)

def psnr_batch(
    noised: torch.Tensor,
    target: torch.Tensor,
    adjust_range: bool = True,
    image_max: float = 1.0,
) -> float:
    """Calculate PSNR (dB) between two batches of arrays."""
    noised = noised.reshape(noised.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    if adjust_range:
        noised_min, _ = torch.min(noised, dim=1)
        noised_max, _ = torch.max(noised, dim=1)
        target_min, _ = torch.min(target, dim=1)
        target_max, _ = torch.max(target, dim=1)

        noised = ((noised.T - noised_min) / (noised_max - noised_min)).T
        target = ((target.T - target_min) / (target_max - target_min)).T

    mse = torch.sum((target - noised) ** 2, dim=1) / noised.shape[1]

    return (
        torch.mean(10 * torch.log10((image_max ** 2) / mse))
    ).item()

def plot_grid(samples: np.ndarray) -> np.ndarray:
    """Plot grid image with given samples. Return the figure as an image."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    for sample, axis in zip(samples, axes.reshape(-1)):
        axis.invert_yaxis()
        axis.get_yaxis().set_ticks([])
        axis.get_xaxis().set_ticks([])
        axis.set_box_aspect(1)

        plot = axis.pcolormesh(sample.squeeze(), cmap="plasma")
        divider = make_axes_locatable(axis)
        clb_ax = divider.append_axes("right", size="5%", pad=0.05)
        clb_ax.set_box_aspect(15)
        plt.colorbar(plot, cax=clb_ax)

    fig.canvas.draw()
    plt.close(fig)

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return image.reshape(fig.canvas.get_width_height()[::-1] + (3,))


# TODO do something with the following functions, prev. used for sidd
def make_patches(data_shape, patch_shape):  # type: ignore # pylint: disable=missing-function-docstring
    patch_shape = np.array(patch_shape)
    data_shape = np.array(data_shape)
    n_patches = np.ceil(data_shape / patch_shape)
    x, y = np.mgrid[  # pylint: disable=invalid-name
        0 : data_shape[0] - patch_shape[0] : n_patches[0] * 1j,
        0 : data_shape[1] - patch_shape[1] : n_patches[1] * 1j,
    ]
    translations = np.round(np.array([x, y]).T.reshape((-1, 2))).astype(int)
    return [
        (slice(t[0], t[0] + patch_shape[0]), slice(t[1], t[1] + patch_shape[1]))
        for t in translations
    ]


def reconstruct_patches(data_patches, patches, data_shape):  # type: ignore # pylint: disable=missing-function-docstring
    data = np.zeros(data_shape, dtype=data_patches.dtype)
    count = np.zeros(data_shape, dtype=data_patches.dtype)
    for data_patch, patch in zip(data_patches, patches):
        data[patch] += data_patch
        count[patch] += 1
    data = data / count
    return data
