"""One layer network with PConv function."""

from typing import Callable, Any
import torch

from .base import BaseNetwork
from .layers.scale_bias import ScaleBias
from .layers.pconv import PConv


class PConvNet(BaseNetwork):
    """Network with one PConv layer."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {
                "filter_size": filter_size,
                "loss_function": loss_function,
                **kwargs,
            }
        )

        self.pconv1 = PConv(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(num_features=1, **kwargs)

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.pconv1(batch)
        batch = self.sb1(batch)

        return batch

class PConvNetDouble(BaseNetwork):
    """Network with one PConv layer."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {
                "filter_size": filter_size,
                "loss_function": loss_function,
                **kwargs,
            }
        )

        self.pconv1 = PConv(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.pconv2 = PConv(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(num_features=1, **kwargs)

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.pconv1(batch)
        batch = self.pconv2(batch)
        batch = self.sb1(batch)

        return batch
