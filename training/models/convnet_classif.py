import torch
import torch.nn as nn
import numpy as np

from utils.scale_bias import ScaleBias

MODEL_NAME = "convnet_classif"


class ConvNetDoubleClassif(nn.Module):
    def __init__(self, filter_size, input_size=(28, 28), **kwargs):
        super(ConvNetDoubleClassif, self).__init__()
        #LeNet5 implem
        nb_classes = 10
        input_size = input_size[0] * input_size[1]
        filter_size=(5, 5)

        self.conv1 = (
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=filter_size,
                      padding=2)
                     .type(kwargs["dtype"])
                     .cuda(kwargs["device"])
        )
        self.conv2 = (
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=filter_size,
                      padding=0)
                     .type(kwargs["dtype"])
                     .cuda(kwargs["device"])
        )

        self.dense1 = nn.Linear(5 * 5 * 16, 120).type(kwargs["dtype"]).cuda(kwargs["device"])
        self.dense2 = nn.Linear(120, 84).type(kwargs["dtype"]).cuda(kwargs["device"])
        self.dense3 = nn.Linear(84, nb_classes).type(kwargs["dtype"]).cuda(kwargs["device"])

        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.dense1(x)
        x = self.sigmoid(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        x = self.dense3(x)

        x = self.sigmoid(x)

        return x


def get_model(model_args):
    model = ConvNetDoubleClassif(**model_args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, verbose=True, threshold=1e-4
    )
    return (
        model,
        opt,
        scheduler,
    )
