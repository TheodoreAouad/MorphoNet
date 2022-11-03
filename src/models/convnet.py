"""One convolution layer network."""

from typing import Union, Tuple, Callable, Any
from torch import nn
from torch import Tensor

from .base import BaseNetwork


class ConvNet(BaseNetwork):
    """Network with one convolution layer."""

    def __init__(
        self,
        filter_size: Union[int, Tuple[int, int]],
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {"filter_size": filter_size, "loss_function": loss_function}
        )

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=filter_size
        ).type(kwargs["dtype"])
        self.relu1 = nn.ReLU()

    def forward(self, batch: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        # pylint: disable=arguments-differ
        batch = self.conv1(batch)
        batch = self.relu1(batch)

        return batch
