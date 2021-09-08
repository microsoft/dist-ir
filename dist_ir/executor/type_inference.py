# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This is the type propagation abstract domain for the abstract interpreter.
Interpreting a DistIR Function using this domain essentially results in type
propagation/inference for the function, assuming inputs are given appropriate
types (from the ir.type module). The resulting abstract state's environment
maps every Value to a type with shape and dtype information, given input types
or example inputs.

(When we say types we also mean shape and device information.)
"""

from typing import Dict, List

from ..ir import Function, FunctionMaker, Op, Value
from ..ir.type import Type, Tensor
from .absint import interpreter
from .type_register import TypePropRegister  # TODO remove this later


def _type_function(function: Function, type_map: Dict[Value, Type]) -> Function:
    """Create a typed version of function, using the types given in type map."""
    new_function = FunctionMaker(name=function.name)
    # A Map from function's values to new_function's (typed) values:
    value_map: Dict[Value, Value] = {}

    # Add inputs to new_function
    for inp in function.inputs:
        new_inp = new_function.add_input_value(inp.name, type_map[inp])
        value_map[inp] = new_inp

    # Duplicate each op, but with types from typed_env
    for op in function.ops:
        # Invariant: inputs of op are already typed (as ops are toposorted)
        typed_inputs = tuple(value_map[inp] for inp in op.inputs)

        # Recursively convert the subfunctions:
        subfunctions = tuple(_type_function(fn, type_map) for fn in op.subfunctions)

        new_op = Op(
            op_type=op.op_type,
            name=op.name,
            inputs=typed_inputs,
            attributes=op.attributes,
            subfunctions=subfunctions,
            output_names=tuple(v.name for v in op.outputs),
            # Look up output types from type_map
            output_types=tuple(type_map[v] for v in op.outputs),
        )
        new_function.ops.append(new_op)

        # Add op's outputs to value_map
        for old_out, out in zip(op.outputs, new_op.outputs):
            value_map[old_out] = out

    return new_function.finalize()


def infer_types(function: Function, inputs: List[Value]) -> Function:
    """Given a function and a list of input values, returns a new function where
    all values are typed.

    inputs: a list/tuple of Values, of the same length as function.inputs, but
    the names are irrelevant.
    """

    def assert_is_typed(v: Value):
        assert v.type is not None
        if isinstance(v.type, Tensor):
            if v.type.shape is None:
                raise ValueError(f"Expected Value {v} to have a shape")

    assert len(inputs) == len(function.inputs)
    for inp in inputs:
        assert_is_typed(inp)

    # Use the type inference AbstractInterpreter to propagate types
    state = interpreter.interpret(function, (v.type for v in inputs))
    type_map = state.env

    return _type_function(function, type_map)
