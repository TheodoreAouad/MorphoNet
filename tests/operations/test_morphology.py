from operations.morphology import Dilation, Erosion
from operations.structuring_elements import Complex
from pytest import fixture
from pathlib import Path
import numpy as np


def load(name):
    return np.load(Path(__file__).parent / "data" / f"{name}.npy")

@fixture
def structuring_element():
    return Complex(filter_size=7, precision="f64")()

# TODO test only func_
def test_dilation(structuring_element):
    image = load("image").reshape(1, 1, 28, 28)
    expected_result = load("dilation").reshape(1, 1, 28, 28)

    np.testing.assert_array_equal(
        expected_result, Dilation(structuring_element)(image, None)
    )

def test_dilation_batch(structuring_element):
    images = load("images").reshape(10, 1, 28, 28)
    expected_result = load("dilations").reshape(10, 1, 28, 28)

    np.testing.assert_array_equal(
        expected_result, Dilation(structuring_element)(images, None)
    )
    
def test_erosion(structuring_element):
    image = load("image_erosion").reshape(1, 1, 28, 28)
    expected_result = load("erosion").reshape(1, 1, 28, 28)

    np.testing.assert_array_equal(
        expected_result, Erosion(structuring_element)(image, None)
    )

def test_erosion_batch(structuring_element):
    images = load("images_erosion").reshape(10, 1, 28, 28)
    expected_result = load("erosions").reshape(10, 1, 28, 28)

    np.testing.assert_array_equal(
        expected_result, Erosion(structuring_element)(images, None)
    )