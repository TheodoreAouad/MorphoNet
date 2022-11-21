"""Init datasets module."""

from .mnist import MNIST
from .fmnist import FashionMNIST

from .base import DataModule

DATASETS = DataModule.listing()
