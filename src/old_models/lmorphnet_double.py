import torch
import torch.nn as nn
import numpy as np

from lmorph import LMorph
from training.models.layers.scale_bias import ScaleBias

MODEL_NAME = "lmorphnet_double"


class LMorphNetDouble(nn.Module):
    def __init__(self, filter_size, **kwargs):
        super(LMorphNetDouble, self).__init__()
        self.lm1 = LMorph(1, 1, filter_size, **kwargs)
        # self.sb1 = ScaleBias(1, **kwargs)

        self.lm2 = LMorph(1, 1, filter_size, **kwargs)
        self.sb1 = ScaleBias(1, **kwargs)

    def forward(self, x):
        x = self.lm1(x)
        # x = self.sb1(x)
        x = self.lm2(x)
        x = self.sb1(x)
        return x


def get_model(model_args):
    model = LMorphNetDouble(**model_args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, verbose=True, threshold=1e-4
    )
    return (
        model,
        opt,
        scheduler,
    )

