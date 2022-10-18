"""Operations package."""

# For subclasses to be available here, and avoid circular imports
from .morphology import (
    Dilation,
    Erosion,
    Opening,
    Closing,
    BDilation,
    BErosion,
    BOpening,
    BClosing,
    WTopHat,
)
from .noise import (
    Salt,
    Pepper,
    SaltPepper,
)
from .base import Operation
OPS = Operation.listing()