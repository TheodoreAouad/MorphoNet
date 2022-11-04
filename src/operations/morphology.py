"""Module containing mathematical morphology related operations."""

from scipy import ndimage
import numpy as np

from .base import MorphologicalOperation

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


class BDilation(MorphologicalOperation):
    """Do a binary dilation on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        crop_h, crop_w = (
            structuring_element.shape[0] // 2,
            structuring_element.shape[1] // 2,
        )

        return ndimage.binary_dilation(
            image > 0, structure=structuring_element
        )[crop_h : image.shape[0] - crop_h, crop_w : image.shape[1] - crop_w]


class BErosion(MorphologicalOperation):
    """Do a binary erosion on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        crop_h, crop_w = (
            structuring_element.shape[0] // 2,
            structuring_element.shape[1] // 2,
        )

        return ndimage.binary_erosion(image > 0, structure=structuring_element)[
            crop_h : image.shape[0] - crop_h, crop_w : image.shape[1] - crop_w
        ]


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
