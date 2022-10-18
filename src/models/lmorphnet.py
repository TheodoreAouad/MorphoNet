"""One layer network with LMorph function."""

from typing import Union, Tuple, Callable, Any
import torch

from .base import BaseNetwork
from .layers.scale_bias import ScaleBias
from .layers.lmorph import LMorph


class LMorphNet(BaseNetwork):
    """Network with one LMorph layer."""

    def __init__(
        self,
        filter_size: Union[int, Tuple[int, int]],
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self.lm1 = LMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(1, **kwargs)

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.lm1(batch)
        batch = self.sb1(batch)

        return batch
