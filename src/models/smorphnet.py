"""One layer network with SMorph function."""

from typing import Callable, Any
import torch

from .base import BaseNetwork
from .layers.scale_bias import ScaleBias
from .layers.smorph import SMorph, SMorphTanh


class SMorphNet(BaseNetwork):
    """Network with one SMorph layer."""

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


class SMorphNetTanh(SMorphNet):
    """Network with one SMorphTanh layer."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(filter_size=filter_size, loss_function=loss_function)
        self.sm1 = SMorphTanh(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(num_features=1, **kwargs)


class SMorphNetDouble(BaseNetwork):
    """Network with two SMorph layers."""

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

        self.sm1 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sm2 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(num_features=1, **kwargs)

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.sm1(batch)
        batch = self.sm2(batch)
        batch = self.sb1(batch)

        return batch
