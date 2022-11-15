"""One layer network with LMorph function."""

from typing import Union, Tuple, Callable, Any
import torch

from .base import BaseNetwork
from .layers.scale_bias import ScaleBias
from .layers.lmorph import LMorph
from .utils import NotTested


class LMorphNet(BaseNetwork):
    """Network with one LMorph layer."""

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


class LMorphNetDouble(BaseNetwork):
    """Network with two LMorph layers."""

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

        self.lm1 = LMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.lm2 = LMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(1, **kwargs)

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.lm1(batch)
        batch = self.lm2(batch)
        batch = self.sb1(batch)

        return batch


# TODO not tested
class LMorphNetFour(BaseNetwork, metaclass=NotTested):
    """Network with four LMorph layers."""

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

        self.lm1 = LMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.lm2 = LMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.lm3 = LMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.lm4 = LMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(1, **kwargs)

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.lm1(batch)
        batch = self.lm2(batch)
        batch = self.lm3(batch)
        batch = self.lm4(batch)
        batch = self.sb1(batch)

        return batch
