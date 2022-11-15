import pytest
from unittest.mock import MagicMock

from losses import Loss
from losses.losses import SSIM, MSE, CrossEntropy

@pytest.mark.parametrize("name, expected_class",
    [
        ("ssim", SSIM),
        ("donotexist", None),
    ]
)
def test_select_(name, expected_class):
    result = Loss.select_(name)

    assert result == expected_class

@pytest.mark.parametrize("name, expected_class",
    [
        ("ssim", SSIM),
        ("mse", MSE),
    ]
)
def test_select(name, expected_class):
    result = Loss.select(name)

    assert type(result) == expected_class

def test_select_neg():
    with pytest.raises(ValueError, match="No matching loss found"):
        _ = Loss.select("donotexist")

def test_listing():
    expected = {
        "ssim",
        "mse",
        "crossentropy",
    }

    assert expected == set(Loss.listing())

@pytest.mark.parametrize("class_",
    [
        SSIM,
        MSE,
        CrossEntropy,
    ]
)
def test_call(class_):
    assert class_()() is not None
