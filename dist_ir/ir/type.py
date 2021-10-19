# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Optional, Sequence, Set, Tuple

import numpy as np

from .device import Device


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

    @staticmethod
    def from_concrete(concrete_value):
        raise NotImplementedError("Each subclass of Type must implement from_concrete")


class Int32(Type):
    """The 32-bit integer type."""

    def __repr__(self):
        return f"Int32[device={self.device}]"

    def size(self):
        return 4

    @staticmethod
    def from_concrete(concrete_value):
        return Int32(concrete_value.device)


class Int64(Type):
    """The 64-bit integer type."""

    def __repr__(self):
        return f"Int64[device={self.device}]"

    def size(self):
        return 8

    @staticmethod
    def from_concrete(concrete_value):
        return Int64(concrete_value.device)


class Float16(Type):
    """The 16-bit float type."""

    def __repr__(self):
        return f"Float16[device={self.device}]"

    def size(self):
        return 2

    @staticmethod
    def from_concrete(concrete_value):
        return Float16(concrete_value.device)


class Float32(Type):
    """The 32-bit float type."""

    def __repr__(self):
        return f"Float32[device={self.device}]"

    def size(self):
        return 4

    @staticmethod
    def from_concrete(concrete_value):
        return Float32(concrete_value.device)


class Float64(Type):
    """The 64-bit float type."""

    def __repr__(self):
        return f"Float64[device={self.device}]"

    def size(self):
        return 8

    @staticmethod
    def from_concrete(concrete_value):
        return Float64(concrete_value.device)


class Bool(Type):
    """The boolean type."""

    def __repr__(self):
        return f"Bool[device={self.device}]"

    def size(self):
        return 1

    @staticmethod
    def from_concrete(concrete_value):
        return Bool(concrete_value.device)


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
        if not isinstance(self.shape, tuple):
            return 0
        return reduce(mul, self.shape) * self.dtype.size()

    @staticmethod
    def from_concrete(concrete_value):
        dtype_to_type = {
            np.int32: Int32,
            np.int64: Int64,
            np.float16: Float16,
            np.float32: Float32,
            np.float64: Float64,
            np.bool: Bool,
        }  # TODO does this map exist/belong somewhere else?
        dtype = dtype_to_type[concrete_value.val.dtype.type](concrete_value.device)
        return Tensor(dtype, concrete_value.val.shape, concrete_value.device)


@dataclass(frozen=True)
class TupleType(Type):

    types: Tuple[Type] = None

    def __init__(self, types):
        # Override __init__ because it doesn't make sense for a tuple to have a
        # device. Devices are stored in each tuple element.
        object.__setattr__(self, "types", types)  # Can't assign to frozen field
        assert isinstance(self.types, tuple)
        assert all(isinstance(t, Type) for t in self.types)
        assert self.device is None

    def __repr__(self):
        elems_str = ", ".join(str(t) for t in self.types)
        return f"Tuple[{elems_str}]"

    def __len__(self):
        return len(self.types)

    def __iter__(self):
        return iter(self.types)

    def get_all_devices(self) -> Set[Device]:
        devices = set()
        for typ in self.types:
            devices.update(typ.get_all_devices())
        return devices

    def size(self):
        size_ = 0.0
        for typ in self.types:
            size_ += typ.size()
        return size_

    @staticmethod
    def from_concrete(concrete_value):
        raise NotImplementedError


def abstract_values(values: Sequence[Any], target_types: Sequence[type]):
    """Abstracts `values` so that they have types `target_types`.

    `values` are values allowed by the abstract interpreter, and `target_types`
    are types allowed by the abstract interpreter (see
    `absint._type_abstraction_graph`).
    """
    return tuple(
        v if isinstance(v, t) else t.from_concrete(v)
        for v, t in zip(values, target_types)
    )
