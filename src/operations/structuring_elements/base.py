"""Base abstract classes to construct structuring elements."""

from abc import abstractmethod, ABCMeta
from typing import Optional, Any, List, Type
import inspect
import logging

from skimage import morphology
import numpy as np

from misc.utils import PRECISIONS_NP
from .utils import shape_aa

#  pylint: disable=abstract-method


class StructuringElement(metaclass=ABCMeta):
    """Abstract class stating required methods for structuring elements."""

    def __init__(self, filter_size: int, precision: str) -> None:
        self.dtype = PRECISIONS_NP[precision]
        self.filter_shape = (filter_size, filter_size)

    @abstractmethod
    def _draw(self, radius: int) -> np.ndarray:
        """Drawing method specific to a structuring element shape."""

    @abstractmethod
    def __call__(self) -> np.ndarray:
        """Method to construct and draw the desired structuring element."""

    @classmethod
    def select_(cls, name: str) -> Optional[Type["StructuringElement"]]:
        """
        Class method iterating over all subclasses to return the desired
        structuring element class.
        """
        if cls.__name__.lower() == name:
            return cls

        for subclass in cls.__subclasses__():
            instance = subclass.select_(name)
            if instance is not None:
                return instance

        return None

    @classmethod
    def select(cls, name: str, **kwargs: Any) -> "StructuringElement":
        """
        Class method iterating over all subclasses to instantiate the desired
        structuring element.
        """

        selected = cls.select_(name)
        if selected is None:
            logging.info("No matching structuring element found")
            return Empty()

        return selected(**kwargs)

    @classmethod
    def listing(cls) -> List[str]:
        """List all the available structuring elements."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__.lower()}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)

    def center_in(
        self,
        center: np.ndarray,
    ) -> np.ndarray:
        """Pad the computed center form to match the desired filter shape."""
        pad_before = (self.filter_shape[0] - center.shape[0]) // 2, (
            self.filter_shape[1] - center.shape[1]
        ) // 2
        pad_after = (
            self.filter_shape[0] - pad_before[0],
            self.filter_shape[1] - pad_before[1],
        )

        centered = np.zeros(shape=self.filter_shape, dtype=self.dtype)
        centered[
            pad_before[0] : pad_after[0], pad_before[1] : pad_after[1]
        ] = center

        return centered


class Empty(StructuringElement):
    """No structuring element."""

    def __init__(self, filter_size: int = -1, precision: str = "") -> None:
        super().__init__(filter_size, precision)

    def _draw(self, radius: int) -> np.ndarray:
        raise NotImplementedError

    def __call__(self) -> np.ndarray:
        raise NotImplementedError


class Disk(StructuringElement):
    """Structuring Element with a disk shape."""

    def _draw(self, radius: int) -> np.ndarray:
        return morphology.disk(radius).astype(self.dtype)


class Diskaa(Disk):
    """Structuring Element with a smoothed disk shape."""

    def _draw(self, radius: int) -> np.ndarray:
        dim = radius * 2 + 1
        res = Disk._draw(self, dim)
        return shape_aa(res, (dim, dim))


class Diamond(StructuringElement):
    """Structuring Element with a diamond shape."""

    def _draw(self, radius: int) -> np.ndarray:
        return morphology.diamond(radius).astype(self.dtype)


class Diamondaa(Diamond):
    """Structuring Element with a smoothed diamond shape."""

    def _draw(self, radius: int) -> np.ndarray:
        dim = radius * 2 + 1
        res = Diamond._draw(self, dim)
        return shape_aa(res, (dim, dim))


class Cross(StructuringElement):
    """Structuring Element with a cross shape."""

    def _draw(self, radius: int) -> np.ndarray:
        res = np.zeros((radius, radius), dtype=self.dtype)
        res[radius // 2, :] = 1.0
        res[:, radius // 2] = 1.0
        return res


class DoubleDisk9(Diskaa, Diamond):
    """Structuring Element with a double disk shape."""

    def __init__(self, filter_size: int, precision: str) -> None:
        super().__init__(filter_size, precision)
        self.filter_shape = (9, 9)
