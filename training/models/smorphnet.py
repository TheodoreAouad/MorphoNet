import torch
import torch.nn as nn
import numpy as np

from smorph import SMorph
from utils.scale_bias import ScaleBias

MODEL_NAME = "smorphnet"

class SMorphNet(nn.Module):
    def __init__(self, filter_size, **kwargs):
        super(SMorphNet, self).__init__()
        self.sm1 = SMorph(1, 1, filter_size, **kwargs)
        self.sb1 = ScaleBias(1, **kwargs)

    def forward(self, x):
        x = self.sm1(x)
        x = self.sb1(x)
        return x


def get_model(model_args):
    model = SMorphNet(**model_args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=20, verbose=True, threshold=1e-4
    )
    return (
        model,
        opt,
        scheduler,
    )

