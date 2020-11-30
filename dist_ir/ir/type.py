from abc import ABC
from functools import reduce
from operator import add, mul
from typing import Generic, Optional, Tuple, TypeVar

from .device import Device
from .utils import singleton

T = TypeVar("T")


class Type:
    def __init__(self, has_device=False):
        self._has_device = has_device
        self._device = None

    def set_device(self, device):
        if self._has_device:
            self._device = device

    def get_all_devices(self):
        if self._has_device and self._device is not None:
            return set([self._device])
        return set()


# TODO might want to have f32, i32 etc instead?


@singleton
class Int(Type):
    """The integer type. A singleton class."""

    def __str__(self):
        return "Int"

    def __repr__(self):
        return str(self)

    @property
    def size(self):
        return 4


@singleton
class Float(Type):
    """The float type. A singleton class."""

    def __str__(self):
        return f"Float"

    def __repr__(self):
        return str(self)

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
        Type.__init__(self, has_device=True)
        self._device = device
        self._shape = shape
        self._dtype = dtype

    def __repr__(self):
        return (
            f"(Tensor(shape={self._shape}, dtype={self._dtype}, device={self._device})"
        )

    def __eq__(self, other):
        return (
            self._dtype == other._dtype
            and self._shape == other._shape
            and self._device == other._device
        )

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

    @property
    def device(self):
        return self._device

    def size(self):
        return reduce(mul, self._shape) * self._dtype.size


class TupleType(Type, Generic[T]):
    def __init__(self, types: Tuple[T]):
        Type.__init__(self)
        self._types = types

    def __str__(self):
        output = "("
        for i in range(len(self._types) - 1):
            output += str(self._types[i]) + ", "
        output += str(self._types[-1]) + ")"
        return output

    def __repr__(self):
        return str(self)

    @property
    def types(self):
        return self._types

    def get_all_devices(self):
        devices = set()
        for typ in self._types:
            devices.update(typ.get_all_devices())
        return devices

    def size(self):
        return reduce(add, [typ.size() for typ in self._types])
