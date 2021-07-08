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

from collections.abc import Sequence
import numpy as np
from typing import Dict, List, Tuple

from ..ir import Device, Function, FunctionMaker, Op, Value
from ..ir.type import Bool, Float32, Int32, Int64, Type, Tensor, TupleType
from .absint import AbstractInterpreter, AbstractState


def _raise_type_error(op, *args):
    raise ValueError(f"Type error: op\n{op}\nwas given arguments\n{tuple(args)}")


# TODO update the below prop functions to be as robust as _allreduce_prop_fn


def _get_dist_ir_dtype_from_numpy_dtype(numpy_dtype, device=None):
    if numpy_dtype == np.int32:
        return Int32(device=device)
    elif numpy_dtype == np.int64:
        return Int64(device=device)
    elif numpy_dtype == np.float32:
        return Float32(device=device)
    else:
        raise NotImplementedError(f"Unsupported numpy dtype {numpy_dtype}")


def _cast_prop_fn(op, x):
    proto_dtype = op.attributes["to"]
    dtype = {
        1: Float32(),
        6: Int32(),
        7: Int64(),
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
    dim = op.attributes["axis"]
    for i, (d0, d1) in enumerate(zip(x.shape, y.shape)):
        if i != dim and d0 != d1:
            _raise_type_error(op, x, y)
    output_shape = tuple(
        n + (y.shape[i] if i == dim else 0) for i, n in enumerate(x.shape)
    )
    return Tensor(dtype=x.dtype, shape=output_shape, device=x.device)


def _constant_prop_fn(op):
    if isinstance(op.attributes["value"], np.ndarray):
        return Tensor(
            shape=op.attributes["value"].shape,
            device=op.attributes["device"],
            dtype=_get_dist_ir_dtype_from_numpy_dtype(op.attributes["value"].dtype),
        )
    else:
        return _get_dist_ir_dtype_from_numpy_dtype(
            op.attributes["value"].dtype, device=op.attributes["device"]
        )


def _constant_of_shape_prop_fn(op, x):
    # TODO: Fix so that x is a constant
    return Tensor(shape=x.shape, device=x.device, dtype=Int32())


def _dropout_prop_fn(op, x, y, z):
    # TODO
    return x


def _elementwise_tensor_op_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and x.dtype == y.dtype
        and x.device == y.device
    ):
        _raise_type_error(op, x, y)
    # Handle broadcasting according to https://numpy.org/doc/stable/user/basics.broadcasting.html.
    shape = []
    for i in range(max(len(x.shape), len(y.shape))):
        x_idx = len(x.shape) - 1 - i
        y_idx = len(y.shape) - 1 - i
        if x_idx >= 0 and y_idx < 0:
            shape.insert(0, x.shape[x_idx])
        elif x_idx < 0 and y_idx >= 0:
            shape.insert(0, y.shape[y_idx])
        elif x.shape[x_idx] >= 1 and y.shape[y_idx] == 1:
            shape.insert(0, x.shape[x_idx])
        elif x.shape[x_idx] == 1 and y.shape[y_idx] >= 1:
            shape.insert(0, y.shape[y_idx])
        elif x.shape[x_idx] == y.shape[y_idx]:
            shape.insert(0, x.shape[x_idx])
        else:
            _raise_type_error(op, x, y)
    return Tensor(shape=tuple(shape), dtype=x.dtype, device=x.device)


def _expand_prop_fn(op, x, y):
    # TODO
    return Tensor(dtype=x.dtype, device=x.device)


def _gather_prop_fn(op, x, y):
    # TODO: Compute the new shape directly instead of using numpy
    # TODO: Fix so that y is a constant
    if not (
        isinstance(x, Tensor)
        and x.shape is not None
        and isinstance(y, Tensor)
        and y.shape is not None
    ):
        _raise_type_error(op, x, y)
    if x.device is None and y.device is None:
        _raise_type_error(op, x, y)
    elif x.device is not None and y.device is None:
        device = x.device
    elif x.device is None and y.device is not None:
        device = y.device
    else:
        if x.device != y.device:
            _raise_type_error(op, x, y)
        device = x.device
    temp = np.zeros(x.shape)
    axis = op.attributes["axis"]
    new_shape = np.take(temp, y.shape, axis=axis).shape
    return Tensor(dtype=x.dtype, shape=new_shape, device=device)


def _identity_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    return x


def _join_prop_fn(op, *xs):
    if not (isinstance(x, Tensor) for x in xs):
        _raise_type_error(op, xs)
    return TupleType(xs)


def _layer_norm_prop_fn(op, x, y, z):
    return Tensor(dtype=x.dtype, device=x.device)


def _loss_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and x.shape == y.shape
        and x.device == y.device
    ):
        _raise_type_error(op, x, y)
    return x


def _loss_grad_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and x.shape == y.shape
        and x.device == y.device
    ):
        _raise_type_error(op, x, y)
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


def _min_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and x.dtype == y.dtype
        and x.device == y.device
    ):
        _raise_type_error(op, x, y)
    return x


def _mpi_allgather_prop_fn(op, *xs):
    devices = tuple(x.device for x in xs)
    dtypes = tuple(x.dtype for x in xs)
    if not (
        all(isinstance(x, Tensor) for x in xs)
        and len(xs) > 0
        and len(set(dtypes)) == 1
        and len(set(devices)) == len(devices)
    ):
        _raise_type_error(op, xs)
    dim = op.attributes["axis"]
    shape = list(xs[0].shape)
    for x in xs[1:]:
        shape[dim] += x.shape[dim]
    return tuple(Tensor(shape=tuple(shape), dtype=dtypes[0], device=d) for d in devices)


def _mpi_allreduce_prop_fn(op, *xs):
    devices = tuple(x.device for x in xs)
    dtypes = tuple(x.dtype for x in xs)
    if not (
        all(isinstance(x, Tensor) for x in xs)
        and len(xs) > 0
        and all(x.shape == xs[0].shape for x in xs)
        and len(set(dtypes)) == 1
        and len(set(devices)) == len(devices)
    ):
        _raise_type_error(op, *xs)
    return xs


def _mpi_allreduce_from_tuple_type_prop_fn(op, xs):
    devices = tuple(t.device for t in xs.types)
    if not (
        isinstance(xs, TupleType)
        and all(isinstance(t, Tensor) for t in xs.types)
        and len(xs.types) > 0
        and all(t.shape == xs.types[0].shape for t in xs.types)
        and len(set(devices)) == len(devices)
    ):
        _raise_type_error(op, xs)
    return xs


def _mpi_broadcast_prop_fn(op, x, to_tuple_type=False):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    devices = op.attributes["devices"]
    if to_tuple_type:
        return TupleType(
            tuple(
                Tensor(dtype=x.dtype, shape=x.shape, device=device)
                for device in devices
            )
        )
    else:
        return tuple(
            Tensor(dtype=x.dtype, shape=x.shape, device=device) for device in devices
        )


def _mpi_broadcast_v2_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    devices = op.attributes["devices"]


def _mpi_gather_prop_fn(op, *xs):
    if not (
        all(isinstance(x, Tensor) for x in xs)
        and len(set(x.shape for x in xs)) == 1
        and len(set(x.shape for x in xs)) == 1
        and len(xs) > 0
    ):
        # TODO: To strictly follow MPI semantics we should check that the output
        # device is not one of the input devices
        _raise_type_error(op, *xs)
    dim = op.attributes["axis"]
    device = op.attributes["device"]
    output_shape = list(xs[0].shape)
    for i in range(1, len(xs)):
        for j in range(len(xs[i].shape)):
            if j == dim:
                output_shape[j] += xs[i].shape[j]
            elif xs[i].shape[j] != xs[0].shape[j]:
                _raise_type_error(op, *xs)
    output_shape = tuple(output_shape)
    return Tensor(dtype=xs[0].dtype, shape=output_shape, device=device)


def _mpi_gather_from_tuple_type_prop_fn(op, x):
    if not (
        isinstance(x, TupleType)
        and all(isinstance(t, Tensor) for t in x.types)
        and len(set(t.shape for t in x.types)) == 1
        and len(set(t.dtype for t in x.types)) == 1
        and len(x.types) > 0
    ):
        # TODO: To strictly follow MPI semantics we should check that the output
        # device is not one of the input devices
        _raise_type_error(op, x)
    dim = op.attributes["axis"]
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


def _mpi_reduce_prop_fn(op, *xs):
    if not (
        all(isinstance(x, Tensor) for x in xs)
        and len(set(x.shape for x in xs)) == 1
        and len(set(x.dtype for x in xs)) == 1
        and len(xs) > 0
    ):
        # TODO: To strictly follow MPI semantics we should check that the output
        # device is not one of the input devices
        _raise_type_error(op, *xs)
    device = op.attributes["device"]
    return Tensor(dtype=xs[0].dtype, shape=xs[0].shape, device=device)


def _mpi_reduce_v2_prop_fn(op, x):
    if not (
        isinstance(x, TupleType)
        and all(isinstance(t, Tensor) for t in x.types)
        and len(set(t.shape for t in x.types)) == 1
        and len(set(t.dtype for t in x.types)) == 1
        and len(x.types) > 0
    ):
        # TODO: To strictly follow MPI semantics we should check that the output
        # device is not one of the input devices
        _raise_type_error(op, x)
    device = op.attributes["device"]
    return Tensor(dtype=x.types[0].dtype, shape=x.types[0].shape, device=device)


def _mpi_scatter_prop_fn(op, x, to_tuple_type=False):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    devices = op.attributes["devices"]
    # Check devices is a list of distinct Devices
    assert isinstance(devices, Sequence) and all(isinstance(d, Device) for d in devices)
    assert len(devices) == len(set(devices))
    dim = op.attributes["axis"]
    # TODO: Should we add another function to raise an attribute error?
    assert dim >= 0 and dim < len(x.shape)
    assert x.shape[dim] % len(devices) == 0
    output_shape = list(x.shape)
    output_shape[dim] //= len(devices)
    output_shape = tuple(output_shape)
    if to_tuple_type:
        return TupleType(
            tuple(
                Tensor(dtype=x.dtype, shape=output_shape, device=device)
                for device in devices
            )
        )
    else:
        return tuple(
            Tensor(dtype=x.dtype, shape=output_shape, device=device)
            for device in devices
        )


def _relu_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(x)
    return x


def _relu_grad_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and x.dtype == y.dtype
        and x.device == y.device
        and x.shape[0] == y.shape[0]
    ):
        _raise_type_error(op, x, y)
    return x
    # return Tensor(dtype=x.dtype, shape=(x.shape[1], y.shape[1]), device=x.device)


def _reshape_prop_fn(op, x, y):
    if not (isinstance(x, Tensor) and isinstance(y, Tensor) and x.device == y.device):
        _raise_type_error(op, x, y)
    return Tensor(device=x.device)


def _select_prop_fn(op, x):
    if not (
        isinstance(x, TupleType)
        and all(isinstance(t, Tensor) for t in x.types)
        and len(x.types) > 0
        and all(t.shape == x.types[0].shape for t in x.types)
        # and len(set(t.device for t in x.types)) == 1
    ):
        _raise_type_error(op, x)
    index = op.attributes["index"]
    return x.types[index]


def _send_prop_fn(op, x):
    device = op.attributes["device"]
    if not isinstance(x, Tensor) or device == x.device:
        _raise_type_error(op, x)
    return Tensor(dtype=x.dtype, shape=x.shape, device=device)


def _shape_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    return x  # Tensor(dtype=Int64(), shape=None, device=x.device)


def _slice_prop_fn(op, x, starts, ends, axes):
    # We don't know the shape of the output, so:
    return Tensor(dtype=x.dtype, shape=None, device=x.device)


def _split_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    num_splits = op.attributes["num_splits"]
    split_dim = op.attributes["axis"]
    output_shape = list(x.shape)
    # TODO: Move this check to attribute error function?
    assert output_shape[split_dim] % num_splits == 0
    output_shape[split_dim] //= num_splits
    output_shape = tuple(output_shape)
    return tuple(
        Tensor(dtype=x.dtype, shape=output_shape, device=x.device)
        for i in range(num_splits)
    )


def _split_v2_prop_fn(op, x):
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
    if not (isinstance(x, Tensor)):
        _raise_type_error(op, x)
    if "perm" in op.attributes:
        perm = op.attributes["perm"]
        if len(perm) != len(x.shape):
            _raise_type_error(op, x)
    else:
        if len(x.shape) != 2:
            _raise_type_error(op, x)
        else:
            perm = (1, 0)
    new_shape = []
    for idx in perm:
        new_shape.append(x.shape[idx])
    return Tensor(dtype=x.dtype, shape=tuple(new_shape), device=x.device)


def _unsqueeze_prop_fn(op, x):
    if not (isinstance(x, Tensor) and x.shape is not None):
        _raise_type_error(op, x)
    axes = op.attributes["axes"]
    shape = list(x.shape)
    new_shape = []
    for i, d in enumerate(shape):
        if i in axes:
            new_shape.append(1)
        new_shape.append(d)
    return Tensor(shape=tuple(new_shape), dtype=x.dtype, device=x.device)


TypePropRegister = {
    ("Add", (Tensor, Tensor)): _elementwise_tensor_op_prop_fn,
    ("Cast", (Tensor,)): _cast_prop_fn,
    # ("Concat", (TupleType,)): _concat_prop_fn,
    ("Concat", (Tensor, Tensor)): _concat_prop_fn,
    ("Constant", ()): _constant_prop_fn,
    ("ConstantOfShape", (Tensor,)): _constant_of_shape_prop_fn,
    ("Div", (Tensor, Tensor)): _elementwise_tensor_op_prop_fn,
    ("Dropout", (Tensor, Tensor, type(Bool()))): _dropout_prop_fn,
    ("Expand", (Tensor, Tensor)): _expand_prop_fn,
    ("Gather", (Tensor, Tensor)): _gather_prop_fn,
    ("Identity", (Tensor,)): _identity_prop_fn,
    (
        "Join",
        (
            Tensor,
            Tensor,
        ),
    ): _join_prop_fn,
    (
        "Join",
        (
            Tensor,
            Tensor,
            Tensor,
            Tensor,
        ),
    ): _join_prop_fn,
    ("MPIAllreduceFromTupleType", (TupleType,)): _mpi_allreduce_from_tuple_type_prop_fn,
    ("MPIAllgather", (Tensor,) * 2): _mpi_allgather_prop_fn,
    ("MPIAllgather", (Tensor,) * 4): _mpi_allgather_prop_fn,
    ("MPIAllgather", (Tensor,) * 8): _mpi_allgather_prop_fn,
    ("MPIAllgather", (Tensor,) * 16): _mpi_allgather_prop_fn,
    ("MPIAllgather", (Tensor,) * 32): _mpi_allgather_prop_fn,
    ("MPIAllgather", (Tensor,) * 64): _mpi_allgather_prop_fn,
    ("MPIAllgather", (Tensor,) * 128): _mpi_allgather_prop_fn,
    ("MPIAllgather", (Tensor,) * 256): _mpi_allgather_prop_fn,
    ("MPIAllgather", (Tensor,) * 512): _mpi_allgather_prop_fn,
    ("MPIAllgather", (Tensor,) * 1024): _mpi_allgather_prop_fn,
    ("MPIAllgather", (Tensor,) * 2048): _mpi_allgather_prop_fn,
    ("MPIAllgather", (Tensor,) * 4096): _mpi_allgather_prop_fn,
    ("MPIAllgather", (Tensor,) * 8192): _mpi_allgather_prop_fn,
    ("MPIAllreduce", (Tensor,) * 2): _mpi_allreduce_prop_fn,
    ("MPIAllreduce", (Tensor,) * 4): _mpi_allreduce_prop_fn,
    ("MPIAllreduce", (Tensor,) * 8): _mpi_allreduce_prop_fn,
    ("MPIAllreduce", (Tensor,) * 16): _mpi_allreduce_prop_fn,
    ("MPIAllreduce", (Tensor,) * 32): _mpi_allreduce_prop_fn,
    ("MPIAllreduce", (Tensor,) * 64): _mpi_allreduce_prop_fn,
    ("MPIAllreduce", (Tensor,) * 128): _mpi_allreduce_prop_fn,
    ("MPIAllreduce", (Tensor,) * 256): _mpi_allreduce_prop_fn,
    ("MPIAllreduce", (Tensor,) * 512): _mpi_allreduce_prop_fn,
    ("MPIAllreduce", (Tensor,) * 1024): _mpi_allreduce_prop_fn,
    ("MPIAllreduce", (Tensor,) * 2048): _mpi_allreduce_prop_fn,
    ("MPIAllreduce", (Tensor,) * 4096): _mpi_allreduce_prop_fn,
    ("MPIAllreduce", (Tensor,) * 8192): _mpi_allreduce_prop_fn,
    ("MPIBroadcast", (Tensor,)): _mpi_broadcast_prop_fn,
    ("MPIBroadcastToTupleType", (Tensor,)): lambda op, x: _mpi_broadcast_prop_fn(
        op, x, True
    ),
    ("MPIGather", (Tensor,) * 2): _mpi_gather_prop_fn,
    ("MPIGather", (Tensor,) * 4): _mpi_gather_prop_fn,
    ("MPIGather", (Tensor,) * 8): _mpi_gather_prop_fn,
    ("MPIGather", (Tensor,) * 16): _mpi_gather_prop_fn,
    ("MPIGather", (Tensor,) * 32): _mpi_gather_prop_fn,
    ("MPIGather", (Tensor,) * 64): _mpi_gather_prop_fn,
    ("MPIGather", (Tensor,) * 128): _mpi_gather_prop_fn,
    ("MPIGather", (Tensor,) * 256): _mpi_gather_prop_fn,
    ("MPIGather", (Tensor,) * 512): _mpi_gather_prop_fn,
    ("MPIGather", (Tensor,) * 1024): _mpi_gather_prop_fn,
    ("MPIGather", (Tensor,) * 2048): _mpi_gather_prop_fn,
    ("MPIGather", (Tensor,) * 4096): _mpi_gather_prop_fn,
    ("MPIGather", (Tensor,) * 8192): _mpi_gather_prop_fn,
    ("MPIGatherFromTupleType", (TupleType,)): _mpi_gather_from_tuple_type_prop_fn,
    ("MPIReduce", (Tensor,) * 2): _mpi_reduce_prop_fn,
    ("MPIReduce", (Tensor,) * 4): _mpi_reduce_prop_fn,
    ("MPIReduce", (Tensor,) * 8): _mpi_reduce_prop_fn,
    ("MPIReduce", (Tensor,) * 16): _mpi_reduce_prop_fn,
    ("MPIReduce", (Tensor,) * 32): _mpi_reduce_prop_fn,
    ("MPIReduce", (Tensor,) * 64): _mpi_reduce_prop_fn,
    ("MPIReduce", (Tensor,) * 128): _mpi_reduce_prop_fn,
    ("MPIReduce", (Tensor,) * 256): _mpi_reduce_prop_fn,
    ("MPIReduce", (Tensor,) * 512): _mpi_reduce_prop_fn,
    ("MPIReduce", (Tensor,) * 1024): _mpi_reduce_prop_fn,
    ("MPIReduce", (Tensor,) * 2048): _mpi_reduce_prop_fn,
    ("MPIReduce", (Tensor,) * 4096): _mpi_reduce_prop_fn,
    ("MPIReduce", (Tensor,) * 8192): _mpi_reduce_prop_fn,
    ("MPIScatter", (Tensor,)): _mpi_scatter_prop_fn,
    ("MPIScatterToTupleType", (Tensor,)): lambda op, x: _mpi_scatter_prop_fn(
        op, x, True
    ),
    ("MPIReduce_v2", (TupleType,)): _mpi_reduce_v2_prop_fn,
    ("Loss", (Tensor, Tensor)): _loss_prop_fn,
    ("LossGrad", (Tensor, Tensor)): _loss_grad_prop_fn,
    ("LayerNormalization", (Tensor, Tensor, Tensor)): _layer_norm_prop_fn,
    ("MatMul", (Tensor, Tensor)): _matmul_prop_fn,
    ("MatMulGrad", (Tensor, Tensor, Tensor)): _matmul_grad_prop_fn,
    ("Min", (Tensor, Tensor)): _min_prop_fn,
    ("Relu", (Tensor,)): _relu_prop_fn,
    ("ReluGrad", (Tensor, Tensor)): _relu_grad_prop_fn,
    ("Reshape", (Tensor, Tensor)): _reshape_prop_fn,
    ("Select", (TupleType,)): _select_prop_fn,
    ("Send", (Tensor,)): _send_prop_fn,
    ("Shape", (Tensor,)): _shape_prop_fn,
    ("SplitDistIR", (Tensor,)): _split_prop_fn,
    ("Split_v2", (Tensor,)): _split_v2_prop_fn,
    # ("Shape", (Tensor,)): TODO
    ("Slice", (Tensor, Tensor, Tensor, Tensor)): _slice_prop_fn,
    ("Sub", (Tensor, Tensor)): _elementwise_tensor_op_prop_fn,
    ("Transpose", (Tensor,)): _transpose_prop_fn,
    ("Unsqueeze", (Tensor,)): _unsqueeze_prop_fn,
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


TypeInferrer = AbstractInterpreter(
    semantics=_create_semantics(TypePropRegister), Tuple=lambda t: TupleType(tuple(t))
)


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
    state = TypeInferrer.interpret(function, (v.type for v in inputs))
    type_map = state.env

    return _type_function(function, type_map)
