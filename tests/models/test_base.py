from models.base import BaseNetwork
from models.smorphnet import SMorphNet
from models.pconvnet import PConvNet

from unittest.mock import MagicMock
import pytest
import torch


@pytest.mark.parametrize(
    "name, expected_class",
    [
        ("smorphnet", SMorphNet),
        ("pconvnet", PConvNet),
        ("donotexist", None),
    ],
)
def test_select_(name, expected_class):
    result = BaseNetwork.select_(name)

    assert result == expected_class


@pytest.mark.parametrize(
    "name, expected_class",
    [
        ("smorphnet", SMorphNet),
        ("pconvnet", PConvNet),
    ],
)
def test_select(name, expected_class):
    result = BaseNetwork.select(
        name, filter_size=MagicMock(), loss_function=MagicMock()
    )

    assert type(result) == expected_class


def test_select_neg():
    with pytest.raises(ValueError, match="The selected model was not found"):
        _ = BaseNetwork.select("donotexist")


def test_listing():
    expected = {
        "smorphnet",
        "lmorphnet",
        "smorphnetdouble",
        "lmorphnetdouble",
        "pconvnet",
        "pconvnetdouble",
        "smorphnettanh",
    }

    assert expected == set(BaseNetwork.listing())


def test_configure_optimizers():
    result = BaseNetwork.configure_optimizers(
        MagicMock(parameters=MagicMock(return_value=[MagicMock(torch.Tensor)]))
    )

    assert "lr_scheduler" in result
    assert "optimizer" in result


def test_validation_step():
    assert 42 == BaseNetwork.validation_step(
        MagicMock(loss_function=MagicMock(return_value=42)),
        (MagicMock(), MagicMock()),
    )


def test_training_step():
    assert 42 == BaseNetwork.training_step(
        MagicMock(loss_function=MagicMock(return_value=42)),
        (MagicMock(), MagicMock()),
    )
