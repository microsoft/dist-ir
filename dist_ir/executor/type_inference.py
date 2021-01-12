"""
A type inference module that converts an untyped DistIR Function into one where
every Value is typed with shape and dtype information, given input types or
example inputs.
"""

from typing import Dict, List

from ..ir import Function, FunctionMaker, Op, Value
from ..ir.type import Type, Tensor, TupleType


def _allreduce_prop_fn(op, x):
    assert isinstance(x, TupleType)
    return x


def _broadcast_prop_fn(op, x):
    assert isinstance(x, Tensor)
    devices = op.attributes["devices"]
    types = []
    for device in devices:
        types.append(Tensor(dtype=x.dtype, shape=x.shape, device=device))
    return TupleType(tuple(types))


def _concat_prop_fn(op, x, y):
    assert isinstance(x, Tensor) and isinstance(y, Tensor)
    assert x.dtype == y.dtype and x.device == y.device
    dim = op.attributes["dim"]
    for i, (d0, d1) in enumerate(zip(x.shape, y.shape)):
        assert i == dim or d0 == d1
    output_shape = list(x.shape)
    output.shape[dim] += y.shape[dim]
    output_shape = tuple(output_shape)
    return Tensor(dtype=x.dtype, shape=output_shape, device=x.device)


def _elementwise_tensor_op_prop_fn(op, x, y):
    assert isinstance(x, Tensor) and isinstance(y, Tensor)
    assert x.dtype == y.dtype and x.shape == y.shape and x.device == y.device
    return x


def _gather_prop_fn(op, x):
    assert isinstance(x, TupleType)
    dim = op.attributes["dim"]
    device = op.attributes["device"]
    output_shape = list(x.types[0].shape)
    for i in range(1, len(x.types)):
        assert len(x.types[i].shape) == len(x.types[0].shape)
        assert x.types[i].dtype == x.types[0].dtype
        for j in range(len(x.types[i].shape)):
            if j == dim:
                output_shape[j] += x.types[i].shape[j]
            else:
                assert x.types[i].shape[j] == x.types[0].shape[j]
    output_shape = tuple(output_shape)
    return Tensor(dtype=x.types[0].dtype, shape=output_shape, device=device)


def _matmul_prop_fn(op, x, y):
    assert isinstance(x, Tensor) and isinstance(y, Tensor)
    assert x.dtype == y.dtype and x.device == y.device
    assert x.shape[1] == y.shape[0]
    return Tensor(dtype=x.dtype, shape=(x.shape[0], y.shape[1]), device=x.device)


def _matmul_grad_prop_fn(op, x, y, z):
    assert isinstance(x, Tensor) and isinstance(y, Tensor) and isinstance(z, Tensor)
    assert x.dtype == y.dtype and x.dtype == z.dtype
    assert x.device == y.device and x.device == z.device
    return (x, y)


def _scatter_prop_fn(op, x):
    assert isinstance(x, Tensor)
    devices = op.attributes["devices"]
    dim = op.attributes["dim"]
    assert x.shape[dim] % len(devices) == 0
    output_shape = list(x.shape)
    output_shape[dim] //= len(devices)
    output_shape = tuple(output_shape)
    types = []
    for device in devices:
        types.append(Tensor(dtype=x.dtype, shape=output_shape, device=device))
    return TupleType(tuple(types))


def _select_prop_fn(op, x):
    assert isinstance(x, TupleType)
    dim = op.attributes["dim"]
    assert isinstance(x.types[dim], Tensor)
    return x.types[dim]


def _send_prop_fn(op, x):
    assert isinstance(x, Tensor)
    device = op.attributes["device"]
    return Tensor(dtype=x.dtype, shape=x.shape, device=device)


def _split_prop_fn(op, x):
    assert isinstance(x, Tensor)
    num_splits = op.attributes["num_splits"]
    split_dim = op.attributes["dim"]
    output_shape = list(x.shape)
    assert output_shape[split_dim] % num_splits == 0
    output_shape[split_dim] //= num_splits
    output_shape = tuple(output_shape)
    types = [
        Tensor(dtype=x.dtype, shape=output_shape, device=x.device)
        for i in range(num_splits)
    ]
    return TupleType(tuple(types))


TypePropRegister = {
    "Add": _elementwise_tensor_op_prop_fn,
    # "Allgather": TODO,
    "Allreduce": _allreduce_prop_fn,
    "Broadcast": _broadcast_prop_fn,
    "Concat": _concat_prop_fn,
    "Gather": _gather_prop_fn,
    "Loss": _elementwise_tensor_op_prop_fn,
    "LossGrad": _elementwise_tensor_op_prop_fn,
    "MatMul": _matmul_prop_fn,
    "MatMulGrad": _matmul_grad_prop_fn,
    "Scatter": _scatter_prop_fn,
    "Select": _select_prop_fn,
    "Send": _send_prop_fn,
    "Split": _split_prop_fn,
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
        if op.op_type == "Pmap":
            # TODO handle Pmaps by getting their input value types from the tuples
            # they map over and then calling infer_types on the subfunction
            raise NotImplementedError
        else:
            out_types = TypePropRegister[op.op_type](op, *input_types)
            if not isinstance(out_types, tuple):
                assert isinstance(out_types, Type)
                out_types = (out_types,)
            subfunctions = []

        new_op = Op(
            op.op_type,
            op.name,
            typed_inputs,
            op.attributes,
            subfunctions,
            tuple(v.name for v in op.out_edges),
            out_types,
        )
        new_function.ops.append(new_op)

        # Add op's outputs to value_map
        for old_out, out in zip(op.out_edges, new_op.out_edges):
            assert_is_typed(out)
            value_map[old_out] = out

    return new_function.finalize()
