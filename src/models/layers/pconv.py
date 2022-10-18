"""Layer implementing the PConv function."""

from typing import Any, Union, Tuple
from torch import nn
from torch.nn import functional as F
import torch
import pytorch_lightning as pl

from .utils import make_pair, init_context, fill_

# TODO check alternate grad for param and filter
# TODO no padding in comparison to lmorph and smorph


class PConv(pl.LightningModule):
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
            init_context(fill_, tensor=self.filter, value=1.0)
            self.p.zero_()

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        with torch.no_grad():
            self.filter.clamp_min_(0.0)

        imin, imax = torch.min(batch).detach(), torch.max(batch).detach()
        batch = 1.0 + (batch - imin) / (imax - imin)

        conv1 = F.conv2d(batch.pow(self.p + 1.0), self.filter)
        conv2 = F.conv2d(batch.pow(self.p), self.filter)

        out = conv1 / conv2

        return out - 1.0
