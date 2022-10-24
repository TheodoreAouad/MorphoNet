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


# TODO check these super calls
class Opening(Dilation, Erosion):
    """Do an opening on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        eroded = super(Erosion, self)._func(image, structuring_element)
        dilated = super(Dilation, self)._func(eroded, structuring_element)
        return dilated


class Closing(Dilation, Erosion):
    """Do a closing on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        dilated = super(Dilation, self)._func(image, structuring_element)
        eroded = super(Erosion, self)._func(dilated, structuring_element)
        return eroded


class WTopHat(Opening):
    """Do a white top-hat on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        opened = super(Opening, self)._func(image, structuring_element)
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
        eroded = super(BErosion, self)._func(image, structuring_element)
        dilated = super(BDilation, self)._func(eroded, structuring_element)
        return dilated


class BClosing(BDilation, BErosion):
    """Do a binary closing on the image."""

    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        dilated = super(BDilation, self)._func(image, structuring_element)
        eroded = super(BErosion, self)._func(dilated, structuring_element)
        return eroded
