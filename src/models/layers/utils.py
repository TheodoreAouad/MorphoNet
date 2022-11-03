"""Utility functions for the layers package."""

from typing import Any, Callable, Union, Tuple
import mlflow
import torch

INIT_PARAM = "tensor_init"


def make_pair(object_: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Make a pair out of the argument, if it is not already a tuple."""
    if isinstance(object_, int):
        return (object_, object_)

    return object_


def init_context(init_function: Callable, **kwargs: Any) -> None:
    """Wrapper around the init function used to log the used function."""
    init_function(**kwargs)
    if mlflow.active_run() is not None:
        mlflow.log_param(INIT_PARAM, init_function.__name__)


def folded_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    fold: float = 0.0,
) -> None:
    """Initialize the tensor in place with a folded normal distribution."""
    tensor.normal_(mean, std).sub_(fold).abs_().add_(fold)


def normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
) -> None:
    """Initialize the tensor in place with a normal distribution."""
    tensor.normal_(mean, std)


def fill_(
    tensor: torch.Tensor,
    value: float,
) -> None:
    """Initialize the tensor in place with given value."""
    tensor.fill_(value)
