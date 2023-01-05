from src.operations.morphology import (
    Dilation,
    Erosion,
    Opening,
    Closing,
    WTopHat,
    BDilation,
    BErosion,
    BClosing,
    BOpening,
)
from src.operations.structuring_elements import Complex, BDiamond
from pytest import fixture
from pathlib import Path
import numpy as np
import torch
from unittest.mock import MagicMock


def path(name):
    return Path(__file__).parent / "data" / f"{name}"


@fixture
def structuring_element():
    return Complex(filter_size=7, precision="f64")


@fixture
def binary_structuring_element():
    return BDiamond(filter_size=7, precision="f64")


def test_dilation(structuring_element):
    image = np.load(path("input_dilation.npy"))
    expected_result = np.load(path("target_dilation.npy"))

    result = Dilation._func(MagicMock(), image, structuring_element())

    torch.testing.assert_close(expected_result, result)


def test_dilation_batch(structuring_element):
    images = torch.load(path("inputs_dilation.pt"))
    expected_result = torch.load(path("targets_dilation.pt"))

    inputs, targets = Dilation(structuring_element)(images, torch.empty(0))

    torch.testing.assert_close(expected_result, targets)
    torch.testing.assert_close(images, inputs)


def test_erosion(structuring_element):
    image = np.load(path("input_erosion.npy"))
    expected_result = np.load(path("target_erosion.npy"))

    result = Erosion._func(MagicMock(), image, structuring_element())

    torch.testing.assert_close(expected_result, result)


def test_erosion_batch(structuring_element):
    images = torch.load(path("inputs_erosion.pt"))
    expected_result = torch.load(path("targets_erosion.pt"))

    inputs, targets = Erosion(structuring_element)(images, torch.empty(0))

    torch.testing.assert_close(expected_result, targets)
    torch.testing.assert_close(images, inputs)


def test_opening(structuring_element):
    images = torch.load(path("inputs_erosion.pt"))
    expected_result = torch.load(path("targets_opening.pt"))

    inputs, targets = Opening(structuring_element)(images, torch.empty(0))

    torch.testing.assert_close(expected_result, targets)
    torch.testing.assert_close(images, inputs)


def test_closing(structuring_element):
    images = torch.load(path("inputs_dilation.pt"))
    expected_result = torch.load(path("targets_closing.pt"))

    inputs, targets = Closing(structuring_element)(images, torch.empty(0))

    torch.testing.assert_close(expected_result, targets)
    torch.testing.assert_close(images, inputs)


def test_wtophat(structuring_element):
    images = torch.load(path("inputs_erosion.pt"))
    expected_result = torch.load(path("targets_wtophat.pt"))

    inputs, targets = WTopHat(structuring_element)(images, torch.empty(0))

    torch.testing.assert_close(expected_result, targets)
    torch.testing.assert_close(images, inputs)


def test_bdilation(binary_structuring_element):
    image = np.load(path("input_bdilation.npy"))
    expected_result = np.load(path("target_bdilation.npy"))

    result = BDilation._func(MagicMock(), image, binary_structuring_element())

    torch.testing.assert_close(expected_result, result)


def test_bdilation_batch(binary_structuring_element):
    images = torch.load(path("inputs_dilation.pt"))
    target_inputs = torch.load(path("target_bdilation_inputs.pt"))
    expected_result = torch.load(path("targets_bdilation.pt"))

    inputs, targets = BDilation(binary_structuring_element)(
        images, torch.empty(0)
    )

    torch.testing.assert_close(expected_result, targets)
    torch.testing.assert_close(target_inputs, inputs)


def test_berosion(binary_structuring_element):
    image = np.load(path("input_berosion.npy"))
    expected_result = np.load(path("target_berosion.npy"))

    result = BErosion._func(MagicMock(), image, binary_structuring_element())

    torch.testing.assert_close(expected_result, result)


def test_berosion_batch(binary_structuring_element):
    images = torch.load(path("inputs_erosion.pt"))
    target_inputs = torch.load(path("target_berosion_inputs.pt"))
    expected_result = torch.load(path("targets_berosion.pt"))

    inputs, targets = BErosion(binary_structuring_element)(
        images, torch.empty(0)
    )

    torch.testing.assert_close(expected_result, targets)
    torch.testing.assert_close(target_inputs, inputs)


def test_bopening(binary_structuring_element):
    images = torch.load(path("inputs_erosion.pt"))
    target_inputs = torch.load(path("target_berosion_inputs.pt"))
    expected_result = torch.load(path("targets_bopening.pt"))

    inputs, targets = BOpening(binary_structuring_element)(
        images, torch.empty(0)
    )

    torch.testing.assert_close(expected_result, targets)
    torch.testing.assert_close(target_inputs, inputs)


def test_bclosing(binary_structuring_element):
    images = torch.load(path("inputs_dilation.pt"))
    target_inputs = torch.load(path("target_bdilation_inputs.pt"))
    expected_result = torch.load(path("targets_bclosing.pt"))

    inputs, targets = BClosing(binary_structuring_element)(
        images, torch.empty(0)
    )

    torch.testing.assert_close(expected_result, targets)
    torch.testing.assert_close(target_inputs, inputs)
