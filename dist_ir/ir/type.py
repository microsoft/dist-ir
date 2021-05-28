from dataclasses import dataclass
from functools import reduce
from operator import add, mul
from typing import Optional, Set, Tuple

from .device import Device
from .utils import singleton
from ..proto import type_pb2


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


class Int32(Type):
    """The 32-bit integer type. A singleton class."""

    def __repr__(self):
        return "Int32"

    @property
    def size(self):
        return 4

    def serialize_to_proto(self):
        return type_pb2.Int32()


@singleton
class Int64(Type):
    """The 64-bit integer type. A singleton class."""

    def __repr__(self):
        return "Int64"

    @property
    def size(self):
        return 8

    def serialize_to_proto(self):
        return type_pb2.Int64()


@singleton
class Float(Type):
    """The float type. A singleton class."""

    def __repr__(self):
        return "Float"

    @property
    def size(self):
        return 4


@singleton
class Bool(Type):
    """The boolean type. A singleton class."""

    def __repr__(self):
        return "Bool"

    @property
    def size(self):
        return 1

    def serialize_to_proto(self):
        return type_pb2.Bool()


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

    def serialize_to_proto(self):
        tensor_proto = type_pb2.Tensor()
        if self.device is not None:
            tensor_proto.device.CopyFrom(self.device.serialize_to_proto())
        else:
            # TODO: Anything?
            pass
        if self.shape is not None:
            for dim in self.shape:
                tensor_proto.shape.append(dim)
        if self.dtype is not None:
            if isinstance(self.dtype, Int32):
                tensor_proto.dtype = type_pb2.DataType.INT32
            elif isinstance(self.dtype, Int64):
                tensor_proto.dtype = type_pb2.DataType.INT64
            elif isinstance(self.dtype, Bool):
                tensor_proto.dtype = type_pb2.DataType.BOOL
            elif isinstance(self.dtype, Float):
                raise NotImplementedError(
                    "Float type will be deprecated in favor of  Float16 and Float32"
                )
            else:
                raise ValueError(f"Unknown dtype {type(self.dtype)}")
        else:
            # TODO: Anything?
            pass
        return tensor_proto


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
        return reduce(add, [typ.size() for typ in self.types])
