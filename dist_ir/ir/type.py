from abc import ABC
from typing import Tuple, Optional

from .utils import singleton


class Type(ABC):
    pass


# TODO might want to have f32, i32 etc instead?


class Primitive(Type):
    """A class to encapsulate primitive data types."""

    def __repr__(self):
        return "Primitive"


@singleton
class Int(Primitive):
    """The integer type. A singleton class."""

    def __repr__(self):
        return "Int"


@singleton
class Float(Primitive):
    """The float type. A singleton class."""

    def __repr__(self):
        return "Float"


class Tensor(Type):
    """The tensor type. Contains a shape and an element type (dtype)."""

    # TODO have a global cache to avoid creating multiple objects of same type?

    def __init__(self, dtype: Type = None, shape: Optional[Tuple[int]] = None):
        self._shape = shape
        assert dtype is None or (
            isinstance(dtype, Type) and not isinstance(dtype, Tensor)
        )
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
