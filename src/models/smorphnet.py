"""One layer network with SMorph function."""

from typing import Union, Tuple, Callable, Any
import torch

from .base import BaseNetwork
from .layers.scale_bias import ScaleBias
from .layers.smorph import SMorph


class SMorphNet(BaseNetwork):
    """Network with one SMorph layer."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self.sm1 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(num_features=1, **kwargs)

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.sm1(batch)
        batch = self.sb1(batch)

        return batch
