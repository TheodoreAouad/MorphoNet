from operations.base import Operation
from operations.morphology import Opening, BErosion
from operations.noise import Salt
from operations.structuring_elements.base import Empty

from unittest.mock import MagicMock
import pytest


@pytest.mark.parametrize(
    "name, expected_class",
    [
        ("opening", Opening),
        ("berosion", BErosion),
        ("salt", Salt),
        ("donotexist", None),
    ],
)
def test_select_(name, expected_class):
    result = Operation.select_(name)

    assert result == expected_class


@pytest.mark.parametrize(
    "name, expected_class",
    [
        ("opening", Opening),
        ("berosion", BErosion),
        ("salt", Salt),
    ],
)
def test_select(name, expected_class):
    result = Operation.select(
        name, structuring_element=MagicMock(), percentage=MagicMock()
    )

    assert type(result) == expected_class


def test_select_neg():
    with pytest.raises(
        ValueError, match="The selected operation was not found"
    ):
        _ = Operation.select("donotexist")


def test_listing():
    expected = {
        "erosion",
        "dilation",
        "opening",
        "closing",
        "berosion",
        "bdilation",
        "bopening",
        "bclosing",
        "wtophat",
        "saltpepper",
        "salt",
        "pepper",
    }

    assert expected == set(Operation.listing())


def test_empty_structuring_element():
    with pytest.raises(
        ValueError, match="Morphological operations need a structuring element."
    ):
        _ = Operation.select("opening", structuring_element=Empty())


def test_empty_structuring_element_binary():
    with pytest.raises(
        ValueError, match="Morphological operations need a structuring element."
    ):
        _ = Operation.select("bopening", structuring_element=Empty())
