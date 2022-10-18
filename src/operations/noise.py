"""Module containing noise related operations."""

from typing import Optional, List, Tuple

import numpy as np

from .base import NoiseOperation


def _random_distribution(
    percentage: int, size: Tuple[int, ...], space: Optional[List[int]] = None
) -> np.ndarray:
    """#TODO"""
    if space is None:
        space = [1]

    nb_total = size[0] * size[1]
    nb_el = int(nb_total * percentage / 100) // len(space)
    arr = np.zeros(nb_total)
    for i, value in enumerate(space):
        start_index = i * nb_el
        arr[start_index : start_index + nb_el] = value

    np.random.shuffle(arr)
    return arr.reshape(size)


class Salt(NoiseOperation):
    """Add salt noise to images."""

    def _func(self, image: np.ndarray, percentage: int) -> np.ndarray:
        rand = _random_distribution(percentage, size=image.shape)
        return np.ma.masked_array(image, rand.reshape(image.shape)).filled(
            np.max(image)
        )


class Pepper(NoiseOperation):
    """Add pepper noise to images."""

    def _func(self, image: np.ndarray, percentage: int) -> np.ndarray:
        rand = _random_distribution(percentage, size=image.shape)
        return np.ma.masked_array(image, rand.reshape(image.shape)).filled(
            np.min(image)
        )


class SaltPepper(NoiseOperation):
    """Add salt and pepper noise to images."""

    def _func(self, image: np.ndarray, percentage: int) -> np.ndarray:
        rand = _random_distribution(percentage, image.shape, [1, 2])
        arr = np.ma.masked_array(
            image, (rand == 1).reshape(image.shape)
        ).filled(0)
        return np.ma.masked_array(arr, (rand == 2).reshape(image.shape)).filled(
            1
        )
