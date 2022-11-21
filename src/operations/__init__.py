"""Operations package."""

# For subclasses to be available here, and avoid circular imports
from .morphology import *
from .noise import *

from .base import Operation

OPS = Operation.listing()
