from abc import abstractmethod
import inspect
# from operations.structuring_elements.base import STRUCTURING_ELEMENTS, StructuringElement

# print(StructuringElement.listing())

# from operations import Operation
# print(Operation.listing())

class initial:
    def __init__(self) -> None:
        self.a = "initial"

    def test(self):
        print(self.a)

class A(initial):
    def __init__(self) -> None:
        self.a = "A"

class B(initial):
    def __init__(self) -> None:
        self.a = "B"

class C(A, B):
    def test(self):
        B.test(self)

A().test()
C().test()