"""Layer implementing the LMorph function."""

from typing import Any, Union, Tuple
from torch import nn
import torch
import pytorch_lightning as pl

from .utils import make_pair, init_context, folded_normal_

PAD_MODE = "reflect"

# TODO check dual lmorph EqMax utility
# TODO check if filter is well clamped at the right time


class LMorph(pl.LightningModule):
    """Module implementing the LMorph function."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Union[int, Tuple[int, int]],
        **kwargs: Any,
    ):
        super().__init__()
        self.filter_shape = make_pair(filter_size)
        self.pad_h = self.filter_shape[0] // 2
        self.pad_w = self.filter_shape[1] // 2
        self.pad = (self.pad_w, self.pad_w, self.pad_h, self.pad_h)

        self.filter = nn.Parameter(
            torch.empty(
                (out_channels, in_channels, *self.filter_shape), **kwargs
            )
        )
        self.p = nn.Parameter(  # pylint: disable=invalid-name
            torch.empty((out_channels, in_channels), **kwargs)
        )

        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize tensors."""
        with torch.no_grad():
            init_context(folded_normal_, tensor=self.filter, mean=0.0, std=0.01)
            self.p.zero_()

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        with torch.no_grad():
            self.filter.clamp_min_(0.0)

        input_padded = nn.functional.pad(batch, self.pad, mode=PAD_MODE)

        imin = input_padded.min().detach()
        imax = input_padded.max().detach()
        input_padded = 1.0 + (input_padded - imin) / (imax - imin)

        raise Exception("Not implemented")
