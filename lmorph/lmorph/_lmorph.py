import torch
from torch import nn


def _pair(s):
    if isinstance(s, int):
        return (s, s)
    return s


def _flatten_batch(tensor):
    n = tensor.size(0)
    return tensor.view(n, -1)


def _init_folded_normal_(tensor: torch.Tensor, mean=0.0, std=1.0, fold=0.0):
    tensor.normal_(mean, std).sub_(fold).abs_().add_(fold)


class EqMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        bmax = x.max(dim).values
        ctx.dim = dim
        ctx.save_for_backward(x.detach(), bmax)
        return bmax

    @staticmethod
    def backward(ctx, grad_y):
        x, bmax = ctx.saved_tensors
        maxes = (x == bmax).type(x.dtype)
        grad_maxes = grad_y * maxes / maxes.sum()
        return grad_maxes, None


class LMorph(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_size,
        dual=False,
        program=None,
        **kwargs,
    ):
        super(LMorph, self).__init__()
        filter_size = _pair(filter_size)
        self.filter = nn.Parameter(
            torch.empty((out_channels, in_channels, *filter_size), **kwargs)
        )
        self.dual = dual

        self.p = nn.Parameter(torch.empty((out_channels, in_channels), **kwargs))

        if dual and in_channels != out_channels:
            raise ValueError(
                f"A dual LMorph layer must have the same number of input and output channels"
            )

        # if program is None:
        #     program = ["all"]
        # self.program = program
        # self.program_idx = 0

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            _init_folded_normal_(self.filter, 0.0, 0.01)
            self.p.zero_()

    def _lmorph(self, input, filter, p):
        imin = input.min().detach()
        imax = input.max().detach()
        input = 1.0 + (input - imin) / (imax - imin)
        return torch.ops.lmorph.lmorph(input, filter, p)

    def forward(self, input):
        out = self._lmorph(input, self.filter, self.p)
        if self.dual:
            fmax = EqMax.apply(_flatten_batch(self.filter), 1).view(
                self.filter.size(0), 1, 1, 1
            )
            out = self._lmorph(out, fmax - self.filter, -self.p)
        return out

    def after_batch(self):
        self.filter.clamp_min_(0.0)

    # def after_epoch(self):
    #     prev_mode = self.program[self.program_idx % len(self.program)]
    #     self.program_idx += 1
    #     next_mode = self.program[self.program_idx % len(self.program)]

    #     if prev_mode != next_mode:
    #         print(f"Switching from {prev_mode} to {next_mode}")

    #         if next_mode == "all":
    #             self.filter.requires_grad_(True)
    #             self.p.requires_grad_(True)
    #         elif next_mode == "filter":
    #             self.filter.requires_grad_(True)
    #             self.p.requires_grad_(False)
    #         elif next_mode == "p":
    #             self.filter.requires_grad_(False)
    #             self.p.requires_grad_(True)
