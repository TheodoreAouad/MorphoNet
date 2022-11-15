import pytest
from unittest.mock import MagicMock

from models.pconvnet import PConvNet, PConvNetDouble

@pytest.mark.parametrize("class_",
    [
        PConvNet,
        PConvNetDouble,
    ]
)
def test_init(class_):
    class_(filter_size=7, loss_function=MagicMock())
