"""Structuring elements package."""

from .structuring_elements import (
    Disk2,
    Diskaa1,
    Diskaa2,
    Diskaa3,
    Diamondaa3,
    Complex,
    Cross3,
    Cross5,
    Cross7,
    Rand,
    BComplex,
    BSquare,
    BDiamond,
    DoubleDisk92,
    DoubleDisk91,
    DoubleDisk70,
    DoubleDisk71,
    Diag,
    ADiag,
    SIADiag,
    IADiag,
)
from .base import StructuringElement

STRUCTURING_ELEMENTS = StructuringElement.listing()
