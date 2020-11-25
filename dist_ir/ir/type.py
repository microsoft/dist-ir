from abc import ABC
from functools import reduce
from operator import mul
from typing import Generic, Optional, Tuple, TypeVar

from .device import Device
from .utils import singleton

T = TypeVar("T")


class Type:
    def __init__(self, device: Optional[Device] = None):
        self._device = device

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device


# TODO might want to have f32, i32 etc instead?


@singleton
class Int(Type):
    """The integer type. A singleton class."""

    def __repr__(self):
        return "Int"

    @property
    def size(self):
        return 4


@singleton
class Float(Type):
    """The float type. A singleton class."""

    def __repr__(self):
        return "Float"

    @property
    def size(self):
        return 4


class Tensor(Type):
    """The tensor type. Contains a shape and an element type (dtype)."""

    # TODO have a global cache to avoid creating multiple objects of same type?

    def __init__(
        self,
        dtype: Type = None,
        shape: Optional[Tuple[int]] = None,
        device: Device = None,
    ):
        Type.__init__(self, device)
        self._shape = shape
        self._dtype = dtype

    def __repr__(self):
        return f"Tensor({self._shape}, {self._dtype})"

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    def size(self):
        return reduce(mul, self._shape)


class ValueTuple(Type, Generic[T]):
    def __init__(self, types: Tuple[T]):
        Type.__init__(self, None)
        self._types = types

    @property
    def types(self):
        return self._types
