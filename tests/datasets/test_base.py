import pytest
from unittest.mock import MagicMock

from datasets import DataModule
from datasets.mnist import MNIST
from datasets.fmnist import FashionMNIST


@pytest.mark.parametrize(
    "name, expected_class",
    [
        ("mnist", MNIST),
        ("donotexist", None),
    ],
)
def test_select_(name, expected_class):
    result = DataModule.select_(name)

    assert result == expected_class


@pytest.mark.parametrize(
    "name, expected_class",
    [
        ("mnist", MNIST),
        ("fashionmnist", FashionMNIST),
    ],
)
def test_select(name, expected_class):
    result = DataModule.select(
        name,
        batch_size=32,
        dataset_path="str",
        precision="f64",
        operation=MagicMock(),
    )

    assert type(result) == expected_class


def test_select_neg():
    with pytest.raises(ValueError, match="The selected dataset was not found"):
        _ = DataModule.select("donotexist")


def test_listing():
    expected = {"fashionmnist", "mnist"}

    assert expected == set(DataModule.listing())
