import pytest
from unittest.mock import MagicMock

from models.smorphnet import SMorphNet, SMorphNetTanh, SMorphNetDouble

@pytest.mark.parametrize("class_",
    [
        SMorphNet,
        SMorphNetTanh,
        SMorphNetDouble,
    ]
)
def test_init(class_):
    class_(filter_size=7, loss_function=MagicMock())
