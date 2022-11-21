"""Module containing mathematical morphology related operations."""

from scipy import ndimage
import numpy as np

from .base import MorphologicalOperation, BinaryMorphologicalOperation

# TODO pad mode must be explicitely the same as for model inputs


class Dilation(MorphologicalOperation):
    """Do a dilation on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        return ndimage.grey_dilation(image, structure=structuring_element)


class Erosion(MorphologicalOperation):
    """Do an erosion on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        return ndimage.grey_erosion(image, structure=structuring_element)


class Opening(Dilation, Erosion):
    """Do an opening on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        eroded = Erosion._func(self, image, structuring_element)
        dilated = Dilation._func(self, eroded, structuring_element)
        return dilated


class Closing(Dilation, Erosion):
    """Do a closing on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        dilated = Dilation._func(self, image, structuring_element)
        eroded = Erosion._func(self, dilated, structuring_element)
        return eroded


class WTopHat(Opening):
    """Do a white top-hat on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        opened = Opening._func(self, image, structuring_element)
        return image - opened


class BDilation(BinaryMorphologicalOperation):
    """Do a binary dilation on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        return ndimage.binary_dilation(image, structure=structuring_element)


class BErosion(BinaryMorphologicalOperation):
    """Do a binary erosion on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        return ndimage.binary_erosion(image, structure=structuring_element)


class BOpening(BDilation, BErosion):
    """Do a binary opening on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        eroded = BErosion._func(self, image, structuring_element)
        dilated = BDilation._func(self, eroded, structuring_element)
        return dilated


class BClosing(BDilation, BErosion):
    """Do a binary closing on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        dilated = BDilation._func(self, image, structuring_element)
        eroded = BErosion._func(self, dilated, structuring_element)
        return eroded
