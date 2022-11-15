"""Utility functions for the models module."""

from abc import ABCMeta
from typing import Any


class NotTested(ABCMeta):
    """
    Class to use as a metaclass to remove the declared class from appearing in
    the __subclasses__() list of a super class (Filtering listing() method
    result).
    """

    def __init__(cls, name: Any, bases: Any, dct: Any) -> None:
        ABCMeta.__init__(cls, name, bases, dct)
        cls.__bases__ = (type("BaseNetwork", (object,), {}),)
