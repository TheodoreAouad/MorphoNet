from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
import pytest
from skimage.util import random_noise
import numpy as np
from pathlib import Path
import torch

from misc.utils import psnr, psnr_batch

def load_image(name):
    return np.asarray(
        Image.open(Path(__file__).parent / "data" / f"{name}.tif")
    )

@pytest.mark.parametrize(
    "name",
    [
        "Barbara",
        "Lena",
        "Baboon",
        "Cameraman",
    ],
)
def test_psnr_uint(name):
    img = load_image(name)
    noised = random_noise(img, mode='salt')

    assert peak_signal_noise_ratio(img, noised, data_range=255) == psnr(
        noised, img, image_max=255, adjust_range=False
    )

@pytest.mark.parametrize(
    "name",
    [
        "Barbara",
        "Lena",
        "Baboon",
        "Cameraman",
    ],
)
def test_psnr_float(name):
    img = load_image(name).astype(np.float64) / 255.0
    noised = random_noise(img, mode='salt')

    assert peak_signal_noise_ratio(img, noised, data_range=1) == psnr(
        noised, img, adjust_range=False,
    )

def test_psnr_batch():
    names = ["Barbara", "Lena", "Baboon"]
    imgs = []
    for name in names:
        imgs.append(load_image(name).astype(np.float64) / 255.0)

    noised = list(map(lambda img: random_noise(img, mode='salt'), imgs))
    psnrs = []
    for img, nimg in zip(imgs, noised):
        psnrs.append(peak_signal_noise_ratio(img, nimg, data_range=1))

    noised = torch.from_numpy(np.array(noised))[:, None, :, :]
    imgs = torch.from_numpy(np.array(imgs))[:, None, :, :]

    assert np.mean(psnrs) == psnr_batch(
        noised, imgs, adjust_range=False,
    )
