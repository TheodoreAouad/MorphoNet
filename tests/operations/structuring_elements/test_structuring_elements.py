from operations.structuring_elements import Complex, Cross3, Cross5, Cross7
from operations.structuring_elements.base import Disk, Diskaa, Diamond, Diamondaa, Empty, DoubleDisk9, StructuringElement
import pytest
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock

from operations.structuring_elements.structuring_elements import Diamondaa3, Diskaa2, Diskaa3, Disk2, Rand, BComplex, BSquare, BDiamond, DoubleDisk92, DoubleDisk91, DoubleDisk70, DoubleDisk71, Diag, ADiag, SIADiag, IADiag, Diskaa1


def get_expected_path(class_, radius):
    return Path(__file__).parent / "data" / f"{class_.__name__.lower()}_{radius}.npy"



@pytest.mark.parametrize("class_, radius",
    [
        (Disk, 1),
        (Disk, 3),
        (Disk, 5),
        (Diskaa, 1),
        (Diskaa, 3),
        (Diskaa, 5),
        (Diamond, 1),
        (Diamond, 3),
        (Diamond, 5),
        (Diamondaa, 1),
        (Diamondaa, 3),
        (Diamondaa, 5),
    ]
)
def test_draw_abstract(class_, radius):
    #TODO assert type also
    path = get_expected_path(class_, radius)
    expected_tructuring_element = np.load(path)
    np.testing.assert_array_equal(
        expected_tructuring_element, class_._draw(MagicMock(dtype=np.float64), radius)
    )


@pytest.mark.parametrize("class_",
    [
        Complex,
        Cross3,
        Cross5,
        Cross7,
        Disk2,
        Diskaa2,
        Diskaa3,
        Diamondaa3,
        BComplex,
        BSquare,
        BDiamond,
        DoubleDisk92,
        DoubleDisk91,
        DoubleDisk70,
        DoubleDisk71,
        Diag,
        ADiag,
        SIADiag,
        IADiag,
        Diskaa1,
    ]
)
def test_instanciate(class_):
    filter_size = 7

    structuring_element = class_(filter_size, "f64")()
    path = get_expected_path(class_, filter_size)
    expected_tructuring_element = np.load(path)
    np.testing.assert_array_equal(
        expected_tructuring_element, structuring_element,
    )

def test_empty():
    structuring_element = Empty()

    with pytest.raises(NotImplementedError):
        structuring_element()

    with pytest.raises(NotImplementedError):
        structuring_element._draw(-1)

    assert structuring_element.filter_shape == (-1, -1)
    assert structuring_element.dtype == "float64"


@pytest.mark.parametrize("name, expected_class, kwargs",
    [
        ("complex", Complex, {"filter_size": 7, "precision": "f64"}),
        ("diskaa2", Diskaa2, {"filter_size": 7, "precision": "f64"}),
    ]
)
def test_select(name, expected_class, kwargs):
    expected = expected_class(**kwargs)
    result = StructuringElement.select(name, **kwargs)

    assert type(expected) == type(result)
    assert expected.dtype == result.dtype
    assert expected.filter_shape == result.filter_shape

    np.testing.assert_array_equal(
        expected(), result()
    )

def test_select_neg():
    expected = Empty()
    result = StructuringElement.select("donotexist")

    assert type(expected) == type(result)
    assert expected.dtype == result.dtype
    assert expected.filter_shape == result.filter_shape

@pytest.mark.parametrize("name, expected_class",
    [
        ("complex", Complex),
        ("donotexist", None),
    ]
)
def test_select_(name, expected_class):
    result = StructuringElement.select_(name)

    assert result == expected_class

def test_listing():
    expected = {
        "disk2",
        "diskaa1",
        "diskaa2",
        "diskaa3",
        "diamondaa3",
        "complex",
        "cross3",
        "cross5",
        "cross7",
        "rand",
        "bcomplex",
        "bsquare",
        "bdiamond",
        "doubledisk92",
        "doubledisk91",
        "doubledisk70",
        "doubledisk71",
        "diag",
        "adiag",
        "siadiag",
        "iadiag",
    }

    assert expected == set(StructuringElement.listing())

def test_rand():
    rand1 = Rand(7, "f64")()
    rand2 = Rand(7, "f64")()

    assert rand1.shape == (7, 7)
    assert rand2.shape == (7, 7)

    with pytest.raises(AssertionError):
        np.testing.assert_equal(rand1, rand2)


def test_doubledisk9():
    class DD9(DoubleDisk9):
        __call__ = MagicMock()

    structuring_element = DD9(-1, "f64")

    assert structuring_element.filter_shape == (9, 9)
    assert structuring_element.dtype == "float64"
