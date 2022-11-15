"""One convolution layer network."""

from typing import Callable, Any
from torch import nn
from torch import Tensor

from .base import BaseNetwork
from .utils import NotTested

# TODO none of them was tested
class ConvNet(BaseNetwork, metaclass=NotTested):
    """Network with one convolution layer."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {"filter_size": filter_size, "loss_function": loss_function}
        )

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=filter_size, **kwargs
        )
        self.relu1 = nn.ReLU()

    def forward(self, batch: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        # pylint: disable=arguments-differ
        batch = self.conv1(batch)
        batch = self.relu1(batch)

        return batch


class ConvNetDouble(BaseNetwork, metaclass=NotTested):
    """Network with two convolution layers."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {"filter_size": filter_size, "loss_function": loss_function}
        )

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=filter_size, **kwargs
        )
        self.conv2 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=filter_size, **kwargs
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, batch: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        # pylint: disable=arguments-differ
        batch = self.conv1(batch)
        batch = self.relu1(batch)
        batch = self.conv2(batch)
        batch = self.relu2(batch)

        return batch


class LeNet5(BaseNetwork, metaclass=NotTested):
    """Implementation of LeNet5 network for classification."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        nb_classes: int = 10,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {"filter_size": filter_size, "loss_function": loss_function}
        )

        filter_shape = (5, 5)

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=filter_shape,
            padding=2,
            **kwargs,
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=filter_shape,
            padding=0,
            **kwargs,
        )

        self.dense1 = nn.Linear(5 * 5 * 16, 120, **kwargs)
        self.dense2 = nn.Linear(120, 84, **kwargs)
        self.dense3 = nn.Linear(84, nb_classes, **kwargs)

        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        # pylint: disable=arguments-differ
        batch = self.conv1(batch)
        batch = self.sigmoid(batch)
        batch = self.pool(batch)

        batch = self.conv2(batch)
        batch = self.sigmoid(batch)
        batch = self.pool(batch)

        batch = batch.view(batch.size(0), -1)

        batch = self.dense1(batch)
        batch = self.sigmoid(batch)
        batch = self.dense2(batch)
        batch = self.sigmoid(batch)
        batch = self.dense3(batch)

        batch = self.sigmoid(batch)

        return batch
