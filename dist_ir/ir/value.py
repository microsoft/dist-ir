from dataclasses import dataclass

from .type import Bool, Int32, Int64, Tensor, Type
from ..proto import value_pb2


@dataclass(frozen=True, eq=False)
class Value:
    """A DistIR value. While values have names, DistIR makes no attempt to ensure
    value names are unique in a function. Therefore Value equality is object
    equality. (TODO correct terminology for this?)
    """

    name: str
    type: Type

    def serialize_to_proto(self):
        value_proto = value_pb2.Value()
        value_proto.name = self.name
        if self.type is not None:
            type_proto = self.type.serialize_to_proto()
            if isinstance(self.type, type(Int32())):
                value_proto.i32.CopyFrom(type_proto)
            elif isinstance(self.type, type(Int64())):
                value_proto.i64.CopyFrom(type_proto)
            elif isinstance(self.type, type(Bool())):
                value_proto.bool.CopyFrom(type_proto)
            elif isinstance(self.type, Tensor):
                value_proto.tensor.CopyFrom(type_proto)
            else:
                raise ValueError(f"Unknown type {type(self.type)}")
        return value_proto
