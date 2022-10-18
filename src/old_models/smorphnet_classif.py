import torch
import torch.nn as nn
import numpy as np

from smorph import SMorph
from training.models.layers.scale_bias import ScaleBias

MODEL_NAME = "smorphnet_classif"


class SMorphNetDoubleClassif(nn.Module):
    lmbda: nn.Parameter

    def __init__(self, filter_size, nb_classes=10, input_size=(28, 28), **kwargs):
        nb_classes = 10
#        input_size = (input_size[0] - 2 * (filter_size - filter_size % 2)) \
#                     * (input_size[1] - 2 * (filter_size - filter_size % 2))
        input_size = 7 * 7
        super(SMorphNetDoubleClassif, self).__init__()
        self.sm11 = SMorph(1, 1, filter_size, **kwargs)
        self.sm12 = SMorph(1, 1, filter_size, **kwargs)
        self.sm21 = SMorph(1, 1, filter_size, **kwargs)
        self.sm22 = SMorph(1, 1, filter_size, **kwargs)
        self.sm31 = SMorph(1, 1, filter_size, **kwargs)
        self.sm32 = SMorph(1, 1, filter_size, **kwargs)
        self.sm41 = SMorph(1, 1, filter_size, **kwargs)
        self.sm42 = SMorph(1, 1, filter_size, **kwargs)
        self.sm51 = SMorph(1, 1, filter_size, **kwargs)
        self.sm52 = SMorph(1, 1, filter_size, **kwargs)
        self.sm61 = SMorph(1, 1, filter_size, **kwargs)
        self.sm62 = SMorph(1, 1, filter_size, **kwargs)
        self.sm71 = SMorph(1, 1, filter_size, **kwargs)
        self.sm72 = SMorph(1, 1, filter_size, **kwargs)
        self.sm81 = SMorph(1, 1, filter_size, **kwargs)
        self.sm82 = SMorph(1, 1, filter_size, **kwargs)

        self.dense1 = nn.Linear(input_size * 8, 120).type(kwargs["dtype"]).cuda(kwargs["device"])
        self.dense2 = nn.Linear(120, 84).type(kwargs["dtype"]).cuda(kwargs["device"])
        self.dense3 = nn.Linear(84, nb_classes).type(kwargs["dtype"]).cuda(kwargs["device"])

        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x1 = self.sm11(input)
        x1 = self.sm12(x1)
        x1 = self.pool(x1)
        x1 = self.pool(x1)

        x2 = self.sm21(input)
        x2 = self.sm22(x2)
        x2 = self.pool(x2)
        x2 = self.pool(x2)

        x3 = self.sm31(input)
        x3 = self.sm32(x3)
        x3 = self.pool(x3)
        x3 = self.pool(x3)

        x4 = self.sm41(input)
        x4 = self.sm42(x4)
        x4 = self.pool(x4)
        x4 = self.pool(x4)

        x5 = self.sm51(input)
        x5 = self.sm52(x5)
        x5 = self.pool(x5)
        x5 = self.pool(x5)

        x6 = self.sm61(input)
        x6 = self.sm62(x6)
        x6 = self.pool(x6)
        x6 = self.pool(x6)

        x7 = self.sm71(input)
        x7 = self.sm72(x7)
        x7 = self.pool(x7)
        x7 = self.pool(x7)

        x8 = self.sm81(input)
        x8 = self.sm82(x8)
        x8 = self.pool(x8)
        x8 = self.pool(x8)
#        x1 = input
#        x2 = input

#        x = x.view(x.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        x5 = x5.view(x5.size(0), -1)
        x6 = x6.view(x6.size(0), -1)
        x7 = x7.view(x7.size(0), -1)
        x8 = x8.view(x8.size(0), -1)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        x = self.dense1(x)
        x = self.sigmoid(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        x = self.dense3(x)

        x = self.sigmoid(x)

        return x

def get_model(model_args):
    model = SMorphNetDoubleClassif(**model_args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=20, verbose=True, threshold=1e-4
    )
    return (
        model,
        opt,
        scheduler,
    )

