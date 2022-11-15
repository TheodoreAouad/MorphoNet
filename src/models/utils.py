from abc import ABCMeta


class NotTested(ABCMeta):
    def __init__(cls, name, bases, dct):
        ABCMeta.__init__(cls, name, bases, dct)
        cls.__bases__ = (type("BaseNetwork", (object,), {}),)
