from operations.noise import _random_distribution, Salt, Pepper, SaltPepper
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import torch

def not_random_distribution():
    return np.array([
        [1, 1, 1, 0, 2],
        [1, 1, 1, 0, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 2, 0, 2],
        [0, 0, 0, 2, 0],
    ])

@pytest.mark.parametrize("percentage, space, nb_expected",
    [
        (0, [1], 0),
        (5, None, 2),
        (20, [1, 2], 5),
        (50, [1], 25),
    ]
)
def test_random_distribution(percentage, space, nb_expected):
    result = _random_distribution(percentage, (10, 5), space)

    assert result.shape == (10, 5)

    for element in (space or [1]):
        assert np.sum(result == element) == nb_expected

def test_pepper_shift():
    image = np.ones((7, 7))
    result, _ = Pepper(0)._func(image, 0)

    assert result.min() >= 0.5
    assert result.max() <= 1.0


@patch("operations.noise._random_distribution", MagicMock(return_value=not_random_distribution()))
@pytest.mark.parametrize("class_, space, nb_expected, neutral",
    [
        (Salt, [1], 12, -3),
        (Pepper, [0], 12, -1),
        (SaltPepper, [0, 1], 6, -1),
    ]
)
def test_noise(class_, space, nb_expected, neutral):
    images = torch.zeros((5, 1, 5, 5)) - 3.0

    inputs, targets = class_(10)(images, images)

    for input_, target in zip(inputs, targets):
        for element in space:
            assert torch.sum(input_ == element).item() == nb_expected

        np.testing.assert_equal(neutral, target.numpy())
