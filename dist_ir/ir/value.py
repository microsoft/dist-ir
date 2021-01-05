from dataclasses import dataclass

from .type import Type


@dataclass(frozen=True, eq=False)
class Value:
    """A DistIR value. While values have names, DistIR makes no attempt to ensure
    value names are unique in a function. Therefore Value equality is object
    equality.
    """

    name: str
    type: Type
