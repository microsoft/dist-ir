from dataclasses import dataclass
from functools import reduce
from operator import add, mul
from typing import Optional, Set, Tuple

from .device import Device
from .utils import singleton


@dataclass(frozen=True)
class Type:
    """Base class for all types."""

    device: Optional[Device] = None

    def get_all_devices(self) -> Set[Device]:
        """Returns all devices that a value of this type lives on. For example,
        a tuple can have different elements live on different devices.

        Subclasses should override this default implementation.
        """
        if self.device is not None:
            return set([self.device])
        return set()


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


@dataclass(frozen=True)
class Tensor(Type):
    """The tensor type. Contains a shape and an element type (dtype)."""

    # TODO have a global cache to avoid creating multiple objects of same type?

    dtype: Type = None  # Unable to make this non-optional, but this should be
    shape: Optional[Tuple[int]] = None

    def __init__(self, dtype=None, shape=None, device=None):
        # TODO make dtype a required argument?
        assert dtype is None or isinstance(dtype, Type)
        assert shape is None or (
            isinstance(shape, tuple) and all(isinstance(n, int) for n in shape)
        )
        object.__setattr__(self, "dtype", dtype)  # Can't assign to frozen field
        object.__setattr__(self, "shape", shape)
        Type.__init__(self, device=device)

    def __repr__(self):
        return f"Tensor[shape={self.shape}, dtype={self.dtype}, device={self.device}]"

    def size(self):
        return reduce(mul, self.shape) * self.dtype.size


@dataclass(frozen=True)
class TupleType(Type):

    types: Tuple[Type] = None

    def __init__(self, types):
        # Override __init__ because it doesn't make sense for a tuple to have a
        # device. Devices are stored in each tuple element.
        object.__setattr__(self, "types", types)  # Can't assign to frozen field
        assert isinstance(types, tuple) and all(isinstance(t, Type) for t in types)
        assert self.device is None

    def __repr__(self):
        elems_str = ", ".join(str(t) for t in self.types)
        return f"Tuple[{elems_str}]"

    def get_all_devices(self) -> Set[Device]:
        devices = set()
        for typ in self.types:
            devices.update(typ.get_all_devices())
        return devices

    def size(self):
        return reduce(add, [typ.size() for typ in self.types])
