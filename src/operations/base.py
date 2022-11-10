"""Base abstract classes to construct target images."""

from abc import abstractmethod, ABCMeta
from typing import List, Any, Optional, Tuple
import inspect
import logging
import sys

import numpy as np
import torch

from misc.utils import fit_nchw
from .structuring_elements import StructuringElement


class Operation(metaclass=ABCMeta):
    """Abstract class stating required methods for operations."""

    @abstractmethod
    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        structuring_element: StructuringElement,
        **kwargs: Any,
    ) -> None:
        try:
            self.structuring_element = structuring_element()
        except NotImplementedError:
            logging.error(
                "Morphological operations need a structuring element."
            )
            sys.exit(1)

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = torch.from_numpy(
            np.array(
                [
                    self._func(x.squeeze(), self.structuring_element)[
                        np.newaxis, ...
                    ]
                    for x in inputs.numpy()
                ]
            )
        )

        return fit_nchw(inputs), fit_nchw(targets)

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
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        results = np.array(
            [self._func(x.squeeze(), self.percentage) for x in inputs.numpy()]
        )

        targets = torch.from_numpy(results[:, 0])
        inputs = torch.from_numpy(results[:, 1])

        return fit_nchw(inputs), fit_nchw(targets)

    @abstractmethod
    def _func(
        self, image: np.ndarray, percentage: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Function returning the clear and noised images."""
