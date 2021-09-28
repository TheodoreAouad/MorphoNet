import torch
from torch import nn


def _pair(s):
    if isinstance(s, int):
        return (s, s)
    return s


class SMorph(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_size,
        dual=False,
        program=None,
        **kwargs,
    ):
        super(SMorph, self).__init__()
        filter_size = _pair(filter_size)
        self.filter = nn.Parameter(
            torch.empty((out_channels, in_channels, *filter_size), **kwargs)
        )
        self.dual = dual

        self.alpha = nn.Parameter(torch.empty((out_channels, in_channels), **kwargs))

        if dual and in_channels != out_channels:
            raise ValueError(
                f"A dual SMorph layer must have the same number of input and output channels"
            )

        # if program is None:
        #     program = ["all"]
        # self.program = program
        # self.program_idx = 0
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.filter.normal_(0.0, 0.01)
            self.alpha.zero_()

    def _smorph_py(self, input, filter, alpha):
        sum = input.view(
            input.size(0), 1, input.size(1), input.size(2), input.size(3), 1, 1
        ) + filter.view(
            filter.size(0), filter.size(1), 1, 1, filter.size(2), filter.size(3)
        )
        sum_alpha = sum * alpha.view(alpha.size(0), alpha.size(1), 1, 1, 1, 1)
        exp_sum_alpha = sum_alpha.exp()
        sum_exp_sum_alpha = sum * exp_sum_alpha

        upper = 0.0
        lower = 0.0
        for fc in range(filter.size(1)):
            for fy in range(filter.size(2)):
                for fx in range(filter.size(3)):
                    ps = sum_exp_sum_alpha[
                        :,
                        :,
                        fc,
                        fy : input.size(2) + 1 - filter.size(2) + fy,
                        fx : input.size(3) + 1 - filter.size(3) + fx,
                        :,
                        :,
                    ].sum((4, 5))
                    upper = upper + ps
                    lower = lower + exp_sum_alpha[
                        :,
                        :,
                        fc,
                        fy : input.size(2) + 1 - filter.size(2) + fy,
                        fx : input.size(3) + 1 - filter.size(3) + fx,
                        :,
                        :,
                    ].sum((4, 5))

        smorph = upper / lower

        return smorph

    def _smorph_py2(self, input, filter, alpha):
        pad_h, pad_w = filter.size(2) // 2, filter.size(3) // 2

        unfolder = nn.Unfold(kernel_size=(filter.size(2), filter.size(3)))
        unfolded = unfolder(input)

        sum = unfolded.transpose(1, 2) + torch.tanh(alpha) * filter.squeeze().ravel()
        sum_alpha = alpha.squeeze() * sum
        exp_sum_alpha = sum_alpha.exp()
        sum_exp_sum_alpha = sum * exp_sum_alpha

        res = sum_exp_sum_alpha.sum(2) / exp_sum_alpha.sum(2)

        return res.view(input.size(0), input.size(1), input.size(2) - 2 * pad_h,
                        input.size(3) - 2 * pad_w)

    def _smorph(self, input, filter, alpha):
        return torch.ops.smorph.smorph(input, filter, alpha)
        #return self._smorph_py2(input, filter, alpha)

    def forward(self, input):
        out = self._smorph(input, self.filter, self.alpha)
        if self.dual:
            out = self._smorph(out, -self.filter, -self.alpha)
        return out

    # def after_epoch(self):
    #     prev_mode = self.program[self.program_idx % len(self.program)]
    #     self.program_idx += 1
    #     next_mode = self.program[self.program_idx % len(self.program)]

    #     if prev_mode != next_mode:
    #         print(f"Switching from {prev_mode} to {next_mode}")

    #         if next_mode == "all":
    #             self.filter.requires_grad_(True)
    #             self.alpha.requires_grad_(True)
    #         elif next_mode == "filter":
    #             self.filter.requires_grad_(True)
    #             self.alpha.requires_grad_(False)
    #         elif next_mode == "p":
    #             self.filter.requires_grad_(False)
    #             self.alpha.requires_grad_(True)
