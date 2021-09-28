import torch
import torch.nn as nn
import numpy as np

from pconv2d.pconv2d import PConv2D
from utils.scale_bias import ScaleBias

MODEL_NAME = "pconvnet_four"


class PConvNet(nn.Module):
    def __init__(self, filter_size, **kwargs):
        super(PConvNet, self).__init__()
        self.pconv1 = PConv2D(1, 1, filter_size, **kwargs)
        self.pconv2 = PConv2D(1, 1, filter_size, **kwargs)
        self.pconv3 = PConv2D(1, 1, filter_size, **kwargs)
        self.pconv4 = PConv2D(1, 1, filter_size, **kwargs)
        self.sb1 = ScaleBias(1, **kwargs)

    def forward(self, x):
        x = self.pconv1(x)
        x = self.pconv2(x)
        x = self.pconv3(x)
        x = self.pconv4(x)
        x = self.sb1(x)
        return x


def get_model(model_args):
    model = PConvNet(**model_args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, verbose=True, threshold=1e-4
    )
    return (
        model,
        opt,
        scheduler,
    )

