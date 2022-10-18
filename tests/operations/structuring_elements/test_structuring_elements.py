from operations.structuring_elements import Complex, Cross3, Cross5, Cross7
from operations.structuring_elements.base import Disk, Diskaa, Diamond, Diamondaa
import pytest
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock

from operations.structuring_elements.structuring_elements import Diamondaa3, Diskaa2, Diskaa3


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
        Diskaa2,
        Diskaa3,
        Diamondaa3,
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
