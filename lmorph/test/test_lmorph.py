import unittest
import torch
from torch.autograd import gradcheck
import lmorph


class TestLMorph(unittest.TestCase):
    def test_forward(self):
        torch.manual_seed(42)

        batch_size = 4
        input_channels = 2
        output_channels = 4
        input_rows = 16
        input_cols = 16
        filter_rows = 7
        filter_cols = 7

        device = torch.device("cuda")

        kwargs = {"dtype": torch.float64, "device": device, "requires_grad": True}

        input = torch.randn(
            batch_size, input_channels, input_rows, input_cols, **kwargs
        )
        input = (input - torch.min(input)) / (torch.max(input) - torch.min(input))
        filter = torch.randn(
            output_channels, input_channels, filter_rows, filter_cols, **kwargs
        ).abs()
        p = torch.randn(output_channels, input_channels, **kwargs)

        variables = [input, filter, p]

        self.assertTrue(gradcheck(torch.ops.lmorph.lmorph, variables))
