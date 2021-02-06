import torch
import torch.nn as nn


class ScaleBias(nn.Module):
    weight: nn.Parameter
    bias: nn.Parameter

    def __init__(self, num_features, **kwargs):
        super(ScaleBias, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features, 1, 1, **kwargs))
        self.bias = nn.Parameter(torch.zeros(num_features, 1, 1, **kwargs))

    def forward(self, input):
        return input * self.weight + self.bias
