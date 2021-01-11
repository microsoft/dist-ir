"""
A type inference module that converts an untyped DistIR Function into one where
every Value is typed with shape and dtype information, given input types or
example inputs.
"""

from typing import Dict, List

from ..ir import Function, FunctionMaker, Op, Value
from ..ir.type import Type, Tensor


def _elementwise_tensor_op_prop_fn(op, x, y):
    assert isinstance(x, Tensor) and isinstance(y, Tensor)
    assert x.dtype == y.dtype and x.shape == y.shape and x.device == y.device
    return x


TypePropRegister = {
    "Add": _elementwise_tensor_op_prop_fn,
}
"""
Type propagation functions:
For each op, a function that returns the types of the outputs of the op,
given the original op and a list of typed input Values.
When we say types we also mean shape and device information.
These functions also perform type checking: that inputs have expected types.
"""


def infer_types(function: Function, inputs: List[Value]) -> Function:
    """Given a function and a list of input values, returns a new function where
    all values are typed.

    inputs: a list/tuple of Values, of the same length as function.inputs, but
    the names are irrelevant.
    """
    new_function = FunctionMaker()
    # A Map from function's values to new_function's (typed) values:
    value_map: Dict[Value, Value] = {}

    def assert_is_typed(v: Value):
        assert v.type is not None
        if isinstance(v.type, Tensor):
            if v.type.shape is None:
                raise ValueError(f"Expected Value {v} to have a shape")

    # Add inputs to new_function
    assert len(inputs) == len(function.inputs)
    for old_inp, inp in zip(function.inputs, inputs):
        assert_is_typed(inp)
        new_inp = new_function.add_input_value(old_inp.name, inp.type)
        value_map[old_inp] = new_inp

    op: Op  # https://stackoverflow.com/q/59102038
    for op in function.ops:
        # Invariant: inputs of op are already typed (as ops are toposorted)
        typed_inputs = tuple(value_map[inp] for inp in op.in_edges)
        input_types = tuple(v.type for v in typed_inputs)

        # Infer types of outputs and create output values
        out_types = TypePropRegister[op.op_type](op, *input_types)
        if not isinstance(out_types, tuple):
            assert isinstance(out_types, Type)
            out_types = (out_types,)

        # TODO Recursively handle subfunctions
        subfunctions = []

        new_function.ops.append(
            Op(
                op.op_type,
                op.name,
                typed_inputs,
                op.attributes,
                subfunctions,
                op.output_names,
                out_types,
            )
        )

    return new_function.finalize()
