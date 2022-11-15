import pytest
from unittest.mock import MagicMock

from models.lmorphnet import LMorphNet, LMorphNetDouble

@pytest.mark.parametrize("class_",
    [
        LMorphNet,
        LMorphNetDouble,
    ]
)
def test_init(class_):
    class_(filter_size=7, loss_function=MagicMock())
