"""Base class for supported losses."""

from abc import ABCMeta, abstractmethod
from typing import Optional, Type, Any, List, Callable
import inspect


class Loss(metaclass=ABCMeta):
    """Abstract class for mainly listing available losses."""

    @abstractmethod
    def __call__(self) -> Callable:
        """Method to instanciate the desired loss function."""

    @classmethod
    def select_(cls, name: str) -> Optional[Type["Loss"]]:
        """
        Class method iterating over all subclasses to return the desired loss
        class.
        """
        if cls.__name__.lower() == name:
            return cls

        for subclass in cls.__subclasses__():
            selected = subclass.select_(name)
            if selected is not None:
                return selected

        return None

    @classmethod
    def select(cls, name: str, **kwargs: Any) -> "Loss":
        """
        Class method iterating over all subclasses to instantiate the desired
        loss.
        """

        selected = cls.select_(name)
        if selected is None:
            raise ValueError("No matching loss found")

        return selected(**kwargs)

    @classmethod
    def listing(cls) -> List[str]:
        """List all the available losses."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__.lower()}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)
