"""
A type inference module that converts an untyped DistIR Function into one where
every Value is typed with shape and dtype information, given input types or
example inputs.

Type inference requires a register mapping ops to type propagation functions:
- This is a function foo(op, x1, x2, .., xN), where op is an N-ary Op, and x1 to
    xN are Types of the inputs.
- The function should check that the inputs have the expected types.
- The function should return the type of the output/a tuple of types of the
    outputs.
(When we say types we also mean shape and device information.)
"""

from typing import Dict, List, Tuple

from ..ir import Device, Function, FunctionMaker, Op, Value
from ..ir.type import Bool, Float, Int32, Int64, Type, Tensor, TupleType


def _raise_type_error(op, *args):
    raise ValueError(f"Type error: op\n{op}\nwas given arguments\n{tuple(args)}")


def _allreduce_prop_fn(op, x):
    devices = tuple(t.device for t in x.types)
    if not (
        isinstance(x, TupleType)
        and all(isinstance(t, Tensor) for t in x.types)
        and len(x.types) > 0
        and all(t.shape == x.types[0].shape for t in x.types)
        and len(set(devices)) == len(devices)
    ):
        _raise_type_error(op, x)
    return x


# TODO update the below prop functions to be as robust as _allreduce_prop_fn


def _broadcast_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    devices = op.attributes["devices"]
    return TupleType(
        tuple(Tensor(dtype=x.dtype, shape=x.shape, device=device) for device in devices)
    )


def _cast_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    dtype = None
    proto_dtype = op.attributes["to"]
    if proto_dtype == 0:
        raise ValueError("Undefined data type")
    elif proto_dtype == 1:
        dtype = Float()
    elif proto_dtype == 6:
        dtype = Int32()
    elif proto_dtype == 7:
        dtype = Int64()
    elif proto_dtype == 9:
        dtype = Bool()
    else:
        raise NotImplementedError(f"Unsupported data type {proto_dtype}")
    return Tensor(dtype=dtype, shape=x.shape, device=x.device)


def _concat_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and x.dtype == y.dtype
        and x.device == y.device
    ):
        _raise_type_error(op, x, y)
    dim = op.attributes["dim"]
    for i, (d0, d1) in enumerate(zip(x.shape, y.shape)):
        if not i != dim and d0 != d1:
            _raise_type_error(op, x, y)
    output_shape = tuple(
        n + (y.shape[i] if i == dim else 0) for i, n in enumerate(x.shape)
    )
    return Tensor(dtype=x.dtype, shape=output_shape, device=x.device)


def _elementwise_tensor_op_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and x.dtype == y.dtype
        and x.shape == y.shape
        and x.device == y.device
    ):
        _raise_type_error(op, x, y)
    return x


def _identity_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    return x


def _matmul_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and x.dtype == y.dtype
        and x.device == y.device
        and x.shape[1] == y.shape[0]
    ):
        _raise_type_error(op, x, y)
    return Tensor(dtype=x.dtype, shape=(x.shape[0], y.shape[1]), device=x.device)


def _matmul_grad_prop_fn(op, x, y, z):
    # TODO: Check that shapes can be multipled together?
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and isinstance(z, Tensor)
        and x.dtype == y.dtype
        and x.dtype == z.dtype
        and x.device == y.device
        and x.device == z.device
    ):
        _raise_type_error(op, x, y, z)

    return (x, y)


def _min_prop_fn(op, *inputs):
    if not (
        all(isinstance(inp, Tensor) for inp in inputs)
        and all(inp.dtype == inputs[0].dtype for inp in inputs)
        and all(inp.shape == inputs[0].shape for inp in inputs)
        and all(inp.device == inputs[0].device for inp in inputs)
    ):
        _raise_type_error(op, inputs)
    return inputs[0]


def _mpi_gather_prop_fn(op, x):
    if not (
        isinstance(x, TupleType)
        and all(isinstance(t, Tensor) for t in x.types)
        and len(set(t.shape for t in x.types)) == 1
        and len(set(t.dtype for t in x.types)) == 1
        and len(x.types) > 0
    ):
        _raise_type_error(op, x)
    dim = op.attributes["dim"]
    device = op.attributes["device"]
    output_shape = list(x.types[0].shape)
    for i in range(1, len(x.types)):
        for j in range(len(x.types[i].shape)):
            if j == dim:
                output_shape[j] += x.types[i].shape[j]
            elif x.types[i].shape[j] != x.types[0].shape[j]:
                _raise_type_error(op, x)
    output_shape = tuple(output_shape)
    return Tensor(dtype=x.types[0].dtype, shape=output_shape, device=device)


def _scatter_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    devices = op.attributes["devices"]
    # Check devices is a list of distinct Devices
    assert isinstance(devices, list) and all(isinstance(d, Device) for d in devices)
    assert len(devices) == len(set(devices))
    dim = op.attributes["dim"]
    # TODO: Should we add another function to raise an attribute error?
    assert x.shape[dim] % len(devices) == 0
    output_shape = list(x.shape)
    output_shape[dim] //= len(devices)
    output_shape = tuple(output_shape)
    return TupleType(
        tuple(
            Tensor(dtype=x.dtype, shape=output_shape, device=device)
            for device in devices
        )
    )


def _select_prop_fn(op, x):
    if not (
        isinstance(x, TupleType)
        and all(isinstance(t, Tensor) for t in x.types)
        and len(x.types) > 0
        and all(t.shape == x.types[0].shape for t in x.types)
        and len(set(t.device for t in x.types)) == 1
    ):
        _raise_type_error(op, x)
    dim = op.attributes["dim"]
    return x.types[dim]


def _send_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    device = op.attributes["device"]
    return Tensor(dtype=x.dtype, shape=x.shape, device=device)


def _shape_prop_fn(op, x):
    if not (isinstance(x, Tensor) and x.shape is not None):
        _raise_type_error(op, x)
    return Tensor(dtype=Int64(), shape=(len(x.shape),), device=x.device)


def _slice_prop_fn(op, x, y, z, xx=None, xy=None):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and isinstance(z, Tensor)
        and (xx is None or isinstance(xx, Tensor))
        and (xy is None or isinstance(xy, Tensor))
    ):
        _raise_type_error(op, x, y, z, xx, xy)
        # TODO: Make y, z, xx, xy constants


def _split_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    num_splits = op.attributes["num_splits"]
    split_dim = op.attributes["dim"]
    output_shape = list(x.shape)
    # TODO: Move this check to attribute error function?
    assert output_shape[split_dim] % num_splits == 0
    output_shape[split_dim] //= num_splits
    output_shape = tuple(output_shape)
    return TupleType(
        tuple(
            Tensor(dtype=x.dtype, shape=output_shape, device=x.device)
            for i in range(num_splits)
        )
    )


def _transpose_prop_fn(op, x):
    # TODO: Support transpose of tensors with > 2 dimensions
    if not (isinstance(x, Tensor) and len(x.shape) == 2):
        _raise_type_error(op, x)
    return Tensor(dtype=x.dtype, shape=x.shape[::-1], device=x.device)


TypePropRegister = {
    "Add": _elementwise_tensor_op_prop_fn,
    # "Allgather": TODO,
    "Allreduce": _allreduce_prop_fn,
    "Broadcast": _broadcast_prop_fn,
    "Cast": _cast_prop_fn,
    "Concat": _concat_prop_fn,
    "Identity": _identity_prop_fn,
    "Loss": _elementwise_tensor_op_prop_fn,
    "LossGrad": _elementwise_tensor_op_prop_fn,
    "MatMul": _matmul_prop_fn,
    "MatMulGrad": _matmul_grad_prop_fn,
    "Min": _min_prop_fn,
    "MPIGather": _mpi_gather_prop_fn,
    "Scatter": _scatter_prop_fn,
    "Select": _select_prop_fn,
    "Send": _send_prop_fn,
    "Shape": _shape_prop_fn,
    "Split": _split_prop_fn,
    "Transpose": _transpose_prop_fn,
}

# Handling pmap specially for now since it needs to return a typed subfunction


def _pmap_prop_fn(op: Op, input_types: Tuple[Type]):
    """pmap maps over a tuple of values, all of the same type and shape, but on
    distinct devices. For convenience, to avoid a lot of zipping, we allow
    multiple inputs, as long as they are all tuples of the same length and the
    list of devices in each tuple are exactly the same.
    """
    # Pmap expects 1 or more tuples as input
    assert isinstance(input_types, tuple) and len(input_types) > 0
    assert all(isinstance(t, TupleType) for t in input_types)
    # Check that pmap's arguments all have same length and shapes
    assert len(set(len(t.types) for t in input_types)) == 1
    for t in input_types:
        assert all(isinstance(x, Tensor) for x in t.types)
        assert len(set(x.shape for x in t.types)) == 1
        assert len(set(x.dtype for x in t.types)) == 1
    # Check that pmap's arguments are on distinct devices
    devices = tuple(x.device for x in input_types[0].types)
    assert len(set(devices)) == len(devices)
    # Check that all inputs have same list of devices
    for t in input_types:
        assert devices == tuple(x.device for x in t.types)

    # Subfunction's inputs are given by pmap's arguments, but on device d
    subfn_input_types = [t.types[0] for t in input_types]

    # Recursively call infer_types on subfunction
    assert len(op.subfunctions) == 1
    subfunctions = [infer_types(op.subfunctions[0], subfn_input_types)]

    # Pmap's output types are given by subfunction's output types
    out_types = tuple(
        TupleType(
            tuple(
                Tensor(shape=t.type.shape, dtype=t.type.dtype, device=d)
                for d in devices
            )
        )
        for t in subfunctions[0].outputs
    )
    return out_types, subfunctions


def infer_types(function: Function, input_types: List[Type]) -> Function:
    """Given a function and a list of input types, returns a new function where
    all values are typed.

    input_types: a list/tuple of Types, of the same length as function.inputs.
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
    assert len(input_types) == len(function.inputs)
    for old_inp, typ in zip(function.inputs, input_types):
        new_inp = new_function.add_input_value(old_inp.name, typ)
        value_map[old_inp] = new_inp

    op: Op  # https://stackoverflow.com/q/59102038
    for op in function.ops:
        # Invariant: inputs of op are already typed (as ops are toposorted)
        typed_inputs = tuple(value_map[inp] for inp in op.inputs)
        input_types = tuple(v.type for v in typed_inputs)

        # Infer types of outputs and create output values
        if op.op_type == "Pmap":
            out_types, subfunctions = _pmap_prop_fn(op, input_types)
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
            tuple(v.name for v in op.outputs),
            out_types,
        )
        new_function.ops.append(new_op)

        # Add op's outputs to value_map
        for old_out, out in zip(op.outputs, new_op.outputs):
            assert_is_typed(out)
            value_map[old_out] = out

    return new_function.finalize()
