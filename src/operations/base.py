"""Base abstract classes to construct target images."""

from abc import abstractmethod, ABCMeta
from typing import List, Any, Optional, Tuple, Type
import inspect
import logging

import numpy as np
import torch
import mlflow

from misc.utils import fit_nchw
from .structuring_elements import StructuringElement

BINARY_THRESHOLD = 0


class Operation(metaclass=ABCMeta):
    """Abstract class stating required methods for operations."""

    def __init__(self) -> None:
        if mlflow.active_run() is not None:
            mlflow.log_param("binary_threshold", BINARY_THRESHOLD)

    @abstractmethod
    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to compute the desired targets."""

    @classmethod
    def select_(cls, name: str) -> Optional[Type["Operation"]]:
        """
        Class method iterating over all subclasses to get the desired operation.
        """
        if cls.__name__.lower() == name:
            return cls

        for subclass in cls.__subclasses__():
            selected = subclass.select_(name)
            if selected is not None:
                return selected

        return None

    @classmethod
    def select(cls, name: str, **kwargs: Any) -> "Operation":
        """
        Class method iterating over all subclasses to instantiate the desired
        operation.
        """
        selected = cls.select_(name)
        if selected is None:
            raise ValueError("The selected operation was not found")

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


class MorphologicalOperation(Operation):
    """Abstract class with code to apply morphological operations."""

    def __init__(  # pylint: disable=unused-argument
        self,
        structuring_element: StructuringElement,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        try:
            self.structuring_element = structuring_element()
        except NotImplementedError as exc:
            raise ValueError(
                "Morphological operations need a structuring element."
            ) from exc

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


class BinaryMorphologicalOperation(Operation):
    """Abstract class with code to apply binary morphological operations."""

    def __init__(  # pylint: disable=unused-argument
        self,
        structuring_element: StructuringElement,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        try:
            self.structuring_element = structuring_element()
        except NotImplementedError as exc:
            raise ValueError(
                "Morphological operations need a structuring element."
            ) from exc

        if len(np.unique(self.structuring_element)) != 2:
            logging.warning(
                "Using non-binary structuring element for binary operation."
            )

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_ = inputs.numpy() > BINARY_THRESHOLD
        targets = torch.from_numpy(
            np.array(
                [
                    self._func(x.squeeze(), self.structuring_element)[
                        np.newaxis, ...
                    ]
                    for x in inputs_
                ]
            )
        )

        return fit_nchw(torch.from_numpy(inputs_).to(inputs.dtype)), fit_nchw(
            targets
        )

    @abstractmethod
    def _func(
        self, image: np.ndarray, structuring_element: np.ndarray
    ) -> np.ndarray:
        """Function returning the processed image."""


# TODO log on mlflow mean psnr / snr
class NoiseOperation(Operation):
    """Abstract class with code to apply noise operations."""

    def __init__(
        self, percentage: int, **kwargs: Any  # pylint: disable=unused-argument
    ) -> None:
        super().__init__()
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
