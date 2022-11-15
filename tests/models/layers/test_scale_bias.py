import pytest
from unittest.mock import MagicMock

from models.layers.scale_bias import ScaleBias


@pytest.mark.parametrize(
    "class_",
    [
        ScaleBias,
    ],
)
def test_init(class_):
    class_(num_features=1)
