"""Init models module."""

from .smorphnet import *
from .lmorphnet import *
from .pconvnet import *
from .convnet import *

from .base import BaseNetwork

MODELS = BaseNetwork.listing()
