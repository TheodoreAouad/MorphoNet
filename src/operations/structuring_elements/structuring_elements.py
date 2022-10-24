"""Implementation of all available structuring elements."""

import numpy as np

from .base import (
    Disk,
    Diskaa,
    DoubleDisk9,
    Diamondaa,
    Cross,
    StructuringElement,
    Diamond,
)

rng = np.random.default_rng()

# TODO test draws


class Disk2(Disk):
    """Construct a disk with radius 2."""

    def __call__(self) -> np.ndarray:
        return self.center_in(self._draw(2))


class Diskaa1(Diskaa):
    """Construct a smoothed disk with radius 1."""

    def __call__(self) -> np.ndarray:
        return self.center_in(self._draw(1))


class Diskaa2(Diskaa):
    """Construct a smoothed disk with radius 2."""

    def __call__(self) -> np.ndarray:
        return self.center_in(self._draw(2))


class Diskaa3(Diskaa):
    """Construct a smoothed disk with radius 3."""

    def __call__(self) -> np.ndarray:
        return self.center_in(self._draw(3))


class Diamondaa3(Diamondaa):
    """Construct a smoothed diamond with radius 3."""

    def __call__(self) -> np.ndarray:
        return self.center_in(self._draw(3))


class Complex(Diskaa, Diamondaa):
    """Construct a complex shape."""

    def __call__(self) -> np.ndarray:
        return self.center_in(Diskaa._draw(self, 3)) - self.center_in(
            Diamondaa._draw(self, 2)
        )


class Cross3(Cross):
    """Construct a cross with side size 3."""

    def __call__(self) -> np.ndarray:
        return self.center_in(self._draw(3))


class Cross5(Cross):
    """Construct a cross with side size 5."""

    def __call__(self) -> np.ndarray:
        return self.center_in(self._draw(5))


class Cross7(Cross):
    """Construct a cross with side size 7."""

    def __call__(self) -> np.ndarray:
        return self.center_in(self._draw(7))


class Rand(StructuringElement):
    """Construct a random structuring element."""

    def _draw(self, radius: int) -> np.ndarray:
        return rng.normal(size=self.filter_shape, loc=1.5, scale=0.05).astype(
            self.dtype
        )

    def __call__(self) -> np.ndarray:
        return self._draw(-1)


class BComplex(Complex):
    """Construct a binary complex shape."""

    def __call__(self) -> np.ndarray:
        return np.where(Complex.__call__(self) < 0.5, 0, 1)


class BSquare(StructuringElement):
    """Construct a binary square shape."""

    def _draw(self, radius: int) -> np.ndarray:
        return np.ones((radius, radius), dtype=self.dtype)

    def __call__(self) -> np.ndarray:
        return self.center_in(self._draw(3))


class BDiamond(StructuringElement):
    """Construct a binary diamond shape."""

    def _draw(self, radius: int) -> np.ndarray:
        linear = np.arange(0, radius * 2 + 1)
        mesh_x, mesh_y = np.meshgrid(linear, linear)
        return np.array(
            np.abs(mesh_x - radius) + np.abs(mesh_y - radius) <= radius,
            dtype=self.dtype,
        )

    def __call__(self) -> np.ndarray:
        return self.center_in(self._draw(3))


# TODO check operator precedence, add parenthesis to make operation easier to read
# TODO check comments and the mentionned sizes
class DoubleDisk92(DoubleDisk9):
    """One disk with size 2 inside another with size 9."""

    def __call__(self) -> np.ndarray:
        return (
            self.center_in(Diskaa._draw(self, 4))
            - self.center_in(Diskaa._draw(self, 3))
            > 0.5
        ) + self.center_in(Diamond._draw(self, 2))


class DoubleDisk91(DoubleDisk9):
    """One disk with size 1 inside another with size 9."""

    def __call__(self) -> np.ndarray:
        return (
            self.center_in(Diskaa._draw(self, 4))
            - self.center_in(Diskaa._draw(self, 3))
            > 0.5
        ) + self.center_in(Diamond._draw(self, 1))


class DoubleDisk70(Diskaa):
    """One disk with size 0 inside another with size 9."""

    def __call__(self) -> np.ndarray:
        self.filter_shape = (7, 7)
        return (
            self.center_in(Diskaa._draw(self, 3))
            - self.center_in(Diskaa._draw(self, 2))
            > 0.5
        )


class DoubleDisk71(Diskaa, Disk):
    """One disk with size 1 inside another with size 9."""

    def __call__(self) -> np.ndarray:
        return (
            self.center_in(Diskaa._draw(self, 3))
            - self.center_in(Diskaa._draw(self, 2))
            > 0.5
        ) + self.center_in(Disk._draw(self, 0))


class Diag(StructuringElement):
    """Left diagonal shape."""

    def _draw(self, radius: int) -> np.ndarray:
        return np.diag([1] * self.filter_shape[0]).astype(self.dtype)

    def __call__(self) -> np.ndarray:
        return self._draw(-1)


class ADiag(StructuringElement):
    """Asymmetric left diagonal shape."""

    def _draw(self, radius: int) -> np.ndarray:
        diag = np.diag([1] * radius)
        diag[3, 4:5] = 1
        diag[1:3, 4] = 1
        diag[1, 3] = 1

        return diag

    def __call__(self) -> np.ndarray:
        return self._draw(7)


class SIADiag(ADiag):
    "Semi-Isolated Asymmetric left diagonal shape (touching only one corner)."

    def _draw(self, radius: int) -> np.ndarray:
        diag = super()._draw(radius)
        diag[0, 0] = 0

        return diag

    def __call__(self) -> np.ndarray:
        return self._draw(7)


class IADiag(SIADiag):
    "Isolated Asymmetric left diagonal shape (not touching any corner)."

    def _draw(self, radius: int) -> np.ndarray:
        diag = super()._draw(radius)
        diag[6, 6] = 0

        return diag

    def __call__(self) -> np.ndarray:
        return self._draw(7)
