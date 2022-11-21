import pytest
from unittest.mock import MagicMock

from models.layers.lmorph import LMorph


@pytest.mark.parametrize(
    "class_",
    [
        LMorph,
    ],
)
def test_init(class_, monkeypatch):
    init_mock = MagicMock()
    monkeypatch.setattr(class_, "init_parameters", init_mock)

    class_(in_channels=1, out_channels=1, filter_size=7)

    init_mock.assert_called()
