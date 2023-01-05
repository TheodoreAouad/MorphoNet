"""Init datasets module."""

from .mnist import MNIST
from .fmnist import FashionMNIST
from .diskorect import Diskorect
from .noisti import NoiSti

from .base import DataModule

DATASETS = DataModule.listing()
