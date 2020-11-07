from abc import ABC
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


@singleton
class Float(Type):
    """The float type. A singleton class."""

    def __repr__(self):
        return "Float"


class Tensor(Type):
    """The tensor type. Contains a shape and an element type (dtype)."""

    # TODO have a global cache to avoid creating multiple objects of same type?

    def __init__(self, dtype: Type, shape: Optional[Tuple[int]] = None):
        self._shape = shape
        assert isinstance(dtype, Type) and not isinstance(dtype, Tensor)
        self._dtype = dtype

    def __repr__(self):
        return f"Tensor({self._shape}, {self._dtype})"

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype
