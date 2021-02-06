import math
from torch import nn
from torch.autograd import Function
import torch
import numpy as np

from utils import pair


def _flatten_batch(tensor):
    n = tensor.size(0)
    return tensor.view(n, -1)


def _init_folded_normal_(tensor: torch.Tensor, mean=0.0, std=1.0, fold=0.0):
    tensor.normal_(mean, std).sub_(fold).abs_().add_(fold)


class PConv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_size,
        dual=False,
        program=None,
        **kwargs,
    ):
        super(PConv2D, self).__init__()
        filter_size = pair(filter_size)
        self.filter = nn.Parameter(
            torch.empty((out_channels, in_channels, *filter_size), **kwargs)
        )
        self.p = nn.Parameter(torch.empty((out_channels, in_channels), **kwargs))
        self.dual = dual

        if program is None:
            program = ["all"]
        self.program = program
        self.program_idx = 0

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # _init_folded_normal_(self.filter)
            # _init_folded_normal_(self.filter, 0.0, 0.01)
            self.filter.fill_(1.0)
            self.p.zero_()

    def _pconv(self, input: torch.Tensor, filter: torch.Tensor, p: torch.Tensor):
        import torch.nn.functional as F

        imin, imax = torch.min(input).detach(), torch.max(input).detach()
        input = 1.0 + (input - imin) / (imax - imin)

        conv1 = F.conv2d(input.pow(p + 1.0), filter)
        conv2 = F.conv2d(input.pow(p), filter)

        out = conv1 / conv2

        return out - 1.0

    def forward(self, input: torch.Tensor):
        out = self._pconv(input, self.filter, self.p)

        if self.dual:
            out = self._pconv(out, self.filter, -self.p)

        return out

    def after_batch(self):
        # Ensure the filter is always positive.
        self.filter.clamp_min_(0.0)

    def after_epoch(self):
        prev_mode = self.program[self.program_idx % len(self.program)]
        self.program_idx += 1
        next_mode = self.program[self.program_idx % len(self.program)]

        if prev_mode != next_mode:
            print(f"Switching from {prev_mode} to {next_mode}")

            if next_mode == "all":
                self.filter.requires_grad_(True)
                self.alpha.requires_grad_(True)
            elif next_mode == "filter":
                self.filter.requires_grad_(True)
                self.alpha.requires_grad_(False)
            elif next_mode == "p":
                self.filter.requires_grad_(False)
                self.alpha.requires_grad_(True)
