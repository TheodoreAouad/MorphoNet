import torch
import torch.nn as nn

from utils.data import flatten_channel


class InstanceRescale(nn.Module):
    min_bound: float
    max_bound: float
    running_min: torch.Tensor
    running_max: torch.Tensor

    def __init__(self, num_features, min_bound=0.0, max_bound=0.0, memory=10, **kwargs):
        super(InstanceRescale, self).__init__()
        self.current_idx = 0
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.register_buffer(
            "running_min", torch.ones(num_features, memory, **kwargs) * float("inf")
        )
        self.register_buffer(
            "running_max", torch.ones(num_features, memory, **kwargs) * -float("inf")
        )

    def forward(self, input):
        with torch.no_grad():
            channels_min = flatten_channel(input).min(2).values.min(0).values
            channels_max = flatten_channel(input).max(2).values.max(0).values
            self.running_min[:, self.current_idx] = channels_min
            self.running_max[:, self.current_idx] = channels_max
            self.current_idx = (self.current_idx + 1) % self.running_min.size(1)

        running_min = self.running_min.min(1).values.reshape((1, -1, 1, 1))
        running_max = self.running_max.max(1).values.reshape((1, -1, 1, 1))

        return self.min_bound + (self.max_bound - self.min_bound) * (
            input - running_min
        ) / (running_max - running_min)
