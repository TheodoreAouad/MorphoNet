"""Layer implementing the SMorph function."""

from typing import Union, Tuple, Any
import torch
from torch import nn
import mlflow
import pytorch_lightning as pl

from .utils import make_pair, init_context, folded_normal_

# TODO shared param, alpha and/or filter
PAD_MODE = "reflect"


class SMorph(pl.LightningModule):
    """Module implementing the SMorph function."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
        **kwargs: Any,
    ):
        super().__init__()
        self.filter_shape = (filter_size, filter_size)
        self.pad_h = self.filter_shape[0] // 2
        self.pad_w = self.filter_shape[1] // 2
        self.pad = (self.pad_w, self.pad_w, self.pad_h, self.pad_h)

        # TODO try pin mem etc. for speed
        self.filter = nn.Parameter(
            torch.empty(
                (out_channels, in_channels, *self.filter_shape), **kwargs
            )
        )
        self.alpha = nn.Parameter(
            torch.empty((out_channels, in_channels), **kwargs)
        )

        self.init_parameters()

        mlflow.log_param("pad_mode", PAD_MODE)

    def init_parameters(self) -> None:
        """Initialize tensors."""
        with torch.no_grad():
            init_context(folded_normal_, tensor=self.filter, mean=0.0, std=0.01)
            self.alpha.zero_()

    # TODO test fold, unfold, function results shapes
    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        input_padded = nn.functional.pad(batch, self.pad, mode=PAD_MODE)

        unfolder_ = nn.Unfold(kernel_size=self.filter_shape)
        unfolded = unfolder_(input_padded)

        sum_ = unfolded.transpose(1, 2) + (
            torch.tanh(self.alpha) * self.filter.squeeze().ravel()
        )
        sum_alpha = self.alpha.squeeze() * sum_
        exp_sum_alpha = sum_alpha.exp()
        sum_exp_sum_alpha = sum_ * exp_sum_alpha

        result = sum_exp_sum_alpha.sum(2) / exp_sum_alpha.sum(2)

        return result.view(*batch.size())
