"""
This is the type propagation abstract domain for the abstract interpreter.
Interpreting a DistIR Function using this domain essentially results in type
propagation/inference for the function, assuming inputs are given appropriate
types (from the ir.type module). The resulting abstract state's environment
maps every Value to a type with shape and dtype information, given input types
or example inputs.

This module contains a register mapping ops to type propagation functions:
- This is a function foo(op, x1, x2, .., xN), where op is an N-ary Op, and x1 to
    xN are Types of the inputs.
- The function doesn't need to check the python types of the inputs
    (e.g. Tensor) as that is given in the register and is checked by the
    abstract interpreter, but it should check that inputs have the expected
    shapes/dtypes.
- The function should return the type of the output/a tuple of types of the
    outputs.
(When we say types we also mean shape and device information.)
"""

from typing import Dict, List, Tuple

from ..ir import Device, Function, FunctionMaker, Op, Value
from ..ir.type import Bool, Float, Int, Type, Tensor, TupleType
from .absint import AbstractInterpreter, AbstractState


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
    proto_dtype = op.attributes["to"]
    dtype = {
        1: Float(),
        6: Int(),
        7: Int(),  # TODO distinguish between int32 and int64
        9: Bool(),
    }[proto_dtype]
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


def _gather_prop_fn(op, x):
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


def _slice_prop_fn(op, x, starts, ends, axes):
    # We don't know the shape of the output, so:
    return Tensor(dtype=x.dtype, shape=None, device=x.device)


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
    ("Add", (Tensor, Tensor)): _elementwise_tensor_op_prop_fn,
    ("Allreduce", (TupleType,)): _allreduce_prop_fn,
    ("Broadcast", (Tensor,)): _broadcast_prop_fn,
    ("Cast", (Tensor,)): _cast_prop_fn,
    ("Concat", (TupleType,)): _concat_prop_fn,
    ("Gather", (TupleType,)): _gather_prop_fn,
    # ("Loss", (Tensor, Tensor)): TODO
    # ("LossGrad", (Tensor, Tensor)): TODO
    ("MatMul", (Tensor, Tensor)): _matmul_prop_fn,
    ("MatMulGrad", (Tensor, Tensor, Tensor)): _matmul_grad_prop_fn,
    # ("Min", (Tensor, Tensor)): TODO
    ("Scatter", (Tensor,)): _scatter_prop_fn,
    ("Select", (Tensor,)): _select_prop_fn,
    ("Send", (Tensor,)): _send_prop_fn,
    ("Split", (Tensor,)): _split_prop_fn,
    # ("Shape", (Tensor,)): TODO
    ("Slice", (Tensor, Tensor, Tensor, Tensor)): _slice_prop_fn,
    ("Transpose", (Tensor,)): _transpose_prop_fn,
}


def _create_semantics(type_prop_register):
    """Creates a semantics for AbstractInterpreter
    (signature -> (state modifiers))
    from a register of type propagation functions
    signature -> (input types -> output types)).
    """

    def convert_impl(type_prop_fn):
        def semantics(op: Op, state: AbstractState):
            # Find the op's inputs in state's environment
            inputs = tuple(state.env[v] for v in op.inputs)

            # Run the type propagation function
            outputs = type_prop_fn(op, *inputs)

            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for x, val in zip(op.outputs, outputs):
                state.env[x] = val

        return semantics

    return {
        signature: convert_impl(type_prop_fn)
        for signature, type_prop_fn in type_prop_register.items()
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
    subfn_inputs = [
        Value(v.name, t.types[0])
        for v, t in zip(op.subfunctions[0].inputs, input_types)
    ]

    # Recursively call infer_types on subfunction
    assert len(op.subfunctions) == 1
    subfunctions = [infer_types(op.subfunctions[0], subfn_inputs)]

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


TypeInferrer = AbstractInterpreter(semantics=_create_semantics(TypePropRegister))


def _type_function(function: Function, type_map: Dict[Value, Type]) -> Function:
    """Create a typed version of function, using the types given in type map."""
    new_function = FunctionMaker()
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
    state = TypeInferrer.interpret(function, (v.type for v in inputs))
    type_map = state.env

    return _type_function(function, type_map)
