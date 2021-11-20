import torch
import torch.nn as nn
import numpy as np

from smorph import SMorph
from utils.scale_bias import ScaleBias

MODEL_NAME = "smorphnet_double"


class SMorphNetDouble(nn.Module):
    lmbda: nn.Parameter

    def __init__(self, filter_size, **kwargs):
        super(SMorphNetDouble, self).__init__()
#        self.sb1 = ScaleBias(1, **kwargs)
        self.sm1 = SMorph(1, 1, filter_size, **kwargs)
#        self.sb2 = ScaleBias(1, **kwargs)
        self.sm2 = SMorph(1, 1, filter_size, **kwargs)
        self.sb3 = ScaleBias(1, **kwargs)
#        self.lmbda = nn.Parameter(torch.empty((1), **kwargs))

    def forward(self, input):
#        x = self.sb1(input)
        x = self.sm1(input)
#        x = self.sb2(x)
        x = self.sm2(x)
#        x = (0.5 + torch.tanh(self.lmbda) / 2) * input - torch.tanh(self.lmbda) * x
        x = self.sb3(x)

        return x

def get_model(model_args):
    model = SMorphNetDouble(**model_args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=20, verbose=True, threshold=1e-4
    )
    return (
        model,
        opt,
        scheduler,
    )

