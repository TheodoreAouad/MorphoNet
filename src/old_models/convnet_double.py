import torch
import torch.nn as nn
import numpy as np

from training.models.layers.scale_bias import ScaleBias

MODEL_NAME = "convnet_double"


class ConvNetDouble(nn.Module):
    def __init__(self, filter_size, **kwargs):
        super(ConvNetDouble, self).__init__()
        self.conv1 = (
            nn.Conv2d(1, 1, filter_size).type(kwargs["dtype"]).cuda(kwargs["device"])
        )
        self.relu1 = nn.ReLU()
        self.conv2 = (
            nn.Conv2d(1, 1, filter_size).type(kwargs["dtype"]).cuda(kwargs["device"])
        )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


def get_model(model_args):
    model = ConvNetDouble(**model_args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, verbose=True, threshold=1e-4
    )
    return (
        model,
        opt,
        scheduler,
    )

