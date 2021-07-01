from collections import Hashable
from frozendict import frozendict
import numpy as np

from ..ir.function import Function, FunctionMaker
from ..ir.op import Op


def sanitize_unhashable_attributes(function):
    """Replaces unhashable op attributes with hashable byte representations.

    Certain attribute values are not hashable (e.g. NumPy ndarrays) so this
    transform constructs a transformed, hashable function without these values.
    This function also returns a map to help restore the replaced values.

    Args:
      function: A DistIR function.

    Returns:
      A DistIR function with fully hashable attributes as well as a map from
      (attribute name, hashable value) -> original (potentially unhashable)
      value.
    """
    assert isinstance(function, Function)
    attribute_map = {}
    value_map = {}
    sanitized_function = FunctionMaker(function.name)
    for inp in function.inputs:
        sanitized_input = sanitized_function.add_input_value(inp.name, inp.type)
        value_map[inp] = sanitized_input
    for op in function.ops:
        inputs = tuple(value_map[inp] for inp in op.inputs)
        sanitized_attributes = {}
        for attr, value in op.attributes.items():
            if isinstance(value, Hashable):
                sanitized_attributes[attr] = value
            else:
                if not isinstance(value, np.ndarray):
                    raise NotImplementedError(
                        f"Unhashable type {type(value)} for op {op.name} "
                        f"attribute {attr}"
                    )
                sanitized_value = value.tobytes()
                sanitized_attributes[attr] = sanitized_value
                attribute_map[(attr, sanitized_value)] = value
            assert isinstance(sanitized_attributes[attr], Hashable)
        new_op = Op(
            op_type=op.op_type,
            name=op.name,
            inputs=inputs,
            attributes=frozendict(sanitized_attributes),
            subfunctions=op.subfunctions,
            output_names=tuple(output.name for output in op.outputs),
            output_types=tuple(output.type for output in op.outputs),
        )
        sanitized_function.ops.append(new_op)
        for orig_output, sanitized_output in zip(op.outputs, new_op.outputs):
            value_map[orig_output] = sanitized_output
    return sanitized_function.finalize(), attribute_map


def restore_unhashable_attributes(function, attribute_map):
    """Undos the sanitized attribute transform by restoring unhashable attributes.

    Args:
      function: An unfinalized DistIR function (FunctionMaker).
      attribute_map: A map from (attribute name, hashable value) ->
      original (potentially unhashable) value.

    Returns:
      An unfinalized DistIR function with the hashable attributes replaced
      with their unhashable original values.
    """
    assert isinstance(function, FunctionMaker)

    restored_function = FunctionMaker(function.name)
    value_map = {}
    for inp in function.inputs:
        restored_input = restored_function.add_input_value(inp.name, inp.type)
        value_map[inp] = restored_input

    for op in function.ops:
        inputs = tuple(value_map[inp] for inp in op.inputs)
        restored_attributes = {}
        if op.attributes is not None:
            for attr, value in op.attributes.items():
                if (attr, value) in attribute_map:
                    restored_attributes[attr] = attribute_map[(attr, value)]
                else:
                    restored_attributes[attr] = value
        new_op = Op(
            op_type=op.op_type,
            name=op.name,
            inputs=inputs,
            attributes=frozendict(restored_attributes),
            subfunctions=op.subfunctions,
            output_names=tuple(output.name for output in op.outputs),
            output_types=tuple(output.type for output in op.outputs),
        )
        restored_function.ops.append(new_op)
        for (output, restored_output) in zip(op.outputs, new_op.outputs):
            value_map[output] = restored_output

    return restored_function
