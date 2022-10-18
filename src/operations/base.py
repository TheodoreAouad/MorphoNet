"""Base abstract classes to construct target images."""

from abc import abstractmethod, ABCMeta
from typing import List, Any, Optional
import inspect

import numpy as np



class Operation(metaclass=ABCMeta):
    """Abstract class stating required methods for operations."""

    @abstractmethod
    def __call__(self, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Method to compute the desired targets."""

    @classmethod
    def select(cls, name: str, **kwargs: Any) -> Optional["Operation"]:
        """
        Class method iterating over all subclasses to instantiate the desired
        operation.
        """
        if cls.__name__.lower() == name:
            return cls(**kwargs)

        for subclass in cls.__subclasses__():
            instance = subclass.select(name, **kwargs)
            if instance is not None:
                return instance

        return None

    @classmethod
    def listing(cls) -> List[str]:
        """List all the available structuring elements."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__.lower()}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)


class MorphologicalOperation(Operation):
    """Abstract class with code to apply morphological operations."""

    def __init__(  # pylint: disable=unused-argument
        self,
        structuring_element: np.ndarray,
        **kwargs: Any,
    ) -> None:
        self.structuring_element = structuring_element

    def __call__(self, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self._func(x.squeeze(), self.structuring_element)[
                    np.newaxis, ...
                ]
                for x in inputs
            ]
        )

    @abstractmethod
    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        """Function returning the processed image."""


class NoiseOperation(Operation):
    """Abstract class with code to apply noise operations."""

    def __init__(
        self, percentage: int, **kwargs: Any  # pylint: disable=unused-argument
    ) -> None:
        self.percentage = percentage

    def __call__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        return np.array(
            [
                self._func(x.squeeze(), self.percentage)[np.newaxis, ...]
                for x in inputs
            ]
        )

    @abstractmethod
    def _func(self, image: np.ndarray, percentage: int) -> np.ndarray:
        """Function returning the noised image."""
