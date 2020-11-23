from abc import ABC
from functools import reduce
from operator import mul
from typing import Tuple, Optional

from .utils import singleton


class Type(ABC):
    pass


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

    def __init__(self, dtype: Type = None, shape: Optional[Tuple[int]] = None):
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
