"""Layer implementing the PConv function."""

from typing import Any, Optional
from torch import nn
from torch.nn import functional as F
import torch
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from models.layers import PAD_MODE
from .utils import make_pair, init_context, fill_
from .base import BaseLayer

# TODO check alternate grad for param and filter
# TODO no padding in comparison to lmorph and smorph


class PConv(BaseLayer):
    """Module implementing the LMorph function."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
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
            init_context(fill_, tensor=self.filter, value=1.0)
            self.p.zero_()

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        with torch.no_grad():
            self.filter.clamp_min_(0.0)

        imin, imax = torch.min(batch).detach(), torch.max(batch).detach()
        input_padded = nn.functional.pad(batch, self.pad, mode=PAD_MODE)

        batch = 1.0 + (input_padded - imin) / (imax - imin)

        conv1 = F.conv2d(batch.pow(self.p + 1.0), self.filter)
        conv2 = F.conv2d(batch.pow(self.p), self.filter)

        out = conv1 / conv2

        return out - 1.0

    def plot_(
        self, axis: Axes, cmap: str = "plasma", comments: Optional[str] = None
    ) -> Axes:
        p = self.p.squeeze().detach().cpu()  # pylint: disable=invalid-name
        if p < 0:
            cmap = "plasma_r"

        axis.invert_yaxis()
        axis.get_yaxis().set_ticks([])
        axis.get_xaxis().set_ticks([])
        axis.set_box_aspect(1)

        plot = axis.pcolormesh(self.filter.squeeze().detach().cpu(), cmap=cmap)
        divider = make_axes_locatable(axis)
        clb_ax = divider.append_axes("right", size="5%", pad=0.05)
        clb_ax.set_box_aspect(15)
        plt.colorbar(plot, cax=clb_ax)

        axis.set_title(r"$p$: " + f"{p:.3f}", fontsize=20)
        if comments is not None:
            axis.set_xlabel(comments, fontsize=20)

        return axis
