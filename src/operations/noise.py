"""Module containing noise related operations."""

from typing import Optional, List, Tuple

import numpy as np

from .base import NoiseOperation


def _random_distribution(
    percentage: int, shape: Tuple[int, ...], space: Optional[List[int]] = None
) -> np.ndarray:
    """
    Create a mask of the desired shape with noisy pixels. Percentage of noise is
    divided between each element of the given space.
    """
    if space is None:
        space = [1]

    nb_total = shape[0] * shape[1]
    nb_el = int(nb_total * percentage / 100) // len(space)
    arr = np.zeros(nb_total)
    for i, value in enumerate(space):
        start_index = i * nb_el
        arr[start_index : start_index + nb_el] = value

    np.random.shuffle(arr)
    return arr.reshape(shape)


class Salt(NoiseOperation):
    """Add salt noise to images."""

    def _func(
        self, image: np.ndarray, percentage: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        rand = _random_distribution(percentage, shape=image.shape)
        return image, np.ma.masked_array(image, rand).filled(1)


class Pepper(NoiseOperation):
    """Add pepper noise to images."""

    def _func(
        self, image: np.ndarray, percentage: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        image = 0.5 + image / 2.0

        rand = _random_distribution(percentage, shape=image.shape)
        return image, np.ma.masked_array(image, rand).filled(0)


class SaltPepper(NoiseOperation):
    """Add salt and pepper noise to images."""

    def _func(
        self, image: np.ndarray, percentage: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        image = 0.5 + image / 2.0

        rand = _random_distribution(percentage, image.shape, [1, 2])
        arr = np.ma.masked_array(
            image, (rand == 1).reshape(image.shape)
        ).filled(0)
        return image, np.ma.masked_array(
            arr, (rand == 2).reshape(image.shape)
        ).filled(1)
