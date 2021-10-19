# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This module contains a register mapping ops to type propagation functions:
- This is a function foo(op, x1, x2, .., xN), where op is an N-ary Op, and x1 to
    xN are Types of the inputs.
- The function doesn't need to check the python types of the inputs
    (e.g. Tensor) as that is given in the register and is checked by the
    abstract interpreter, but it should check that inputs have the expected
    shapes/dtypes.
- The function should return the type of the output/a tuple of types of the
    outputs.
- The inputs and outputs of these functions are objects that are instances of
    (subclasses of) ir.Type.
"""

from collections.abc import Sequence
import numpy as np

from ..ir import Device
from ..ir.type import Bool, Float32, Int32, Int64, Tensor, TupleType


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


def _concat_prop_fn(op, *xs):
    if not all(
        isinstance(x, Tensor) and x.dtype == xs[0].dtype and x.device == xs[0].device
        for x in xs
    ):
        _raise_type_error(op, *xs)
    dim = op.attributes["axis"]
    for i, ds in enumerate(zip(x.shape for x in xs)):
        if i != dim and any(d != ds[0] for d in ds):
            _raise_type_error(op, *xs)
    new_dim = sum(x.shape[dim] for x in xs)
    output_shape = tuple(new_dim if i == dim else n for i, n in enumerate(xs[0].shape))
    return Tensor(dtype=xs[0].dtype, shape=output_shape, device=xs[0].device)


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
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and x.shape is not None
        and y.shape is not None
        and x.device is not None
        and y.device is not None
        and x.device == y.device
    ):
        _raise_type_error(op, x, y)
    if "axis" in op.attributes:
        axis = op.attributes["axis"]
    else:
        axis = 0
    if axis != 0:
        raise NotImplementedError("abstract Gather function only supports axis 0")

    new_shape = y.shape + x.shape[1:]
    return Tensor(shape=new_shape, device=x.device, dtype=x.dtype)


def _gemm_prop_fn(op, x, y, z):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and isinstance(z, Tensor)
        and x.dtype == y.dtype
        and x.dtype == z.dtype
        and x.device == y.device
        and x.device == z.device
        and x.shape[1] == y.shape[0]
        and len(z.shape) == 1
        and z.shape[0] == y.shape[1]
    ):
        _raise_type_error(op, x, y, z)
    return Tensor(shape=(x.shape[0], y.shape[1]), dtype=x.dtype, device=x.device)


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


def _loss_prop_fn(op, x, y, n):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and isinstance(n, Int32)
        and x.shape == y.shape
        and x.device == y.device
        and x.device == n.device
    ):
        _raise_type_error(op, x, y, n)
    return x


def _loss_grad_prop_fn(op, x, y, n):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and isinstance(n, Int32)
        and x.shape == y.shape
        and x.device == y.device
        and x.device == n.device
    ):
        _raise_type_error(op, x, y)
    return x


def _matmul_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and type(x.dtype) == type(y.dtype)
        and x.device == y.device
        and len(x.shape) == len(y.shape)
        and x.shape[len(x.shape) - 1] == y.shape[len(y.shape) - 2]
    ):
        _raise_type_error(op, x, y)
    new_shape = list(x.shape[:-2])
    new_shape.append(x.shape[len(x.shape) - 2])
    new_shape.append(y.shape[len(y.shape) - 1])
    return Tensor(dtype=x.dtype, shape=tuple(new_shape), device=x.device)


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
    dtypes = tuple(type(x.dtype) for x in xs)
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
    return tuple(Tensor(shape=tuple(shape), dtype=x.dtype, device=x.device) for x in xs)


def _mpi_allreduce_prop_fn(op, *xs):
    devices = tuple(x.device for x in xs)
    dtypes = tuple(type(x.dtype) for x in xs)
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
    devices = op.attributes["devices"]
    if isinstance(x, Tensor):
        tuple_ = tuple(
            Tensor(dtype=x.dtype, shape=x.shape, device=device) for device in devices
        )
    elif isinstance(x, Int32):
        tuple_ = tuple(Int32(device=device) for device in devices)
    else:
        _raise_type_error(op, x)
    if to_tuple_type:
        return TupleType(tuple_)
    else:
        return tuple_


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


def _mul_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and x.shape == y.shape
        and x.dtype == y.dtype
        and x.device == y.device
    ):
        _raise_type_error(op, x, y)
    return x


def _reduce_mean_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    if "keepdims" in op.attributes:
        keepdims = op.attributes["keepdims"]
    else:
        keepdims = 1
    axis = set(tuple(op.attributes["axes"]))
    output_shape = []
    for i in range(len(x.shape)):
        j = len(x.shape) - i - 1
        reduce_dim = j in axis or (j == len(x.shape) - 1 and -1 in axis)
        if not reduce_dim:
            output_shape.append(x.shape[j])
        elif keepdims:
            output_shape.append(1)
    output_shape.reverse()
    return Tensor(shape=tuple(output_shape), dtype=x.dtype, device=x.device)


def _relu_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(x)
    return x


def _relu_grad_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, Tensor)
        and type(x.dtype) == type(y.dtype)
        and x.device == y.device
        and x.shape[0] == y.shape[0]
    ):
        _raise_type_error(op, x, y)
    return x


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
    if device == x.device:
        _raise_type_error(op, x)
    if isinstance(x, Tensor):
        if x.dtype is None:
            _raise_type_error(op, x)
        dtype = type(x.dtype)(device=device)
        return Tensor(dtype=dtype, shape=x.shape, device=device)
    elif isinstance(x, Int32):
        return Int32(device=device)
    else:
        raise_type_error(op, x)


def _sgd_prop_fn(op, *xs):
    weights = xs[: (len(xs) // 2)]
    gradients = xs[(len(xs) // 2) :]
    for (w, dw) in zip(weights, gradients):
        if not (
            isinstance(w, Tensor)
            and isinstance(dw, Tensor)
            and w.shape == dw.shape
            and w.device == dw.device
        ):
            _raise_type_error(op, weights, gradients)
    return weights


def _split_prop_fn(op, x):
    axis = op.attributes["axis"]
    split = op.attributes["split"]
    sections = []
    n = 0
    for s in split[:-1]:
        sections.append(n + s)
        n += s
    sections.append(x.shape[axis])
    output_types = []
    prev_section = 0
    for section in sections:
        output_shape = []
        for i in range(axis):
            output_shape.append(x.shape[i])
        output_shape.append(section - prev_section)
        for i in range(axis + 1, len(x.shape)):
            output_shape.append(x.shape[i])
        prev_section = section
        output_types.append(
            Tensor(shape=tuple(output_shape), device=x.device, dtype=x.dtype)
        )
    return tuple(output_types)


def _split_uniform_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    num_splits = op.attributes["num_splits"]
    split_dim = op.attributes["axis"]
    output_shape = list(x.shape)
    # TODO: Move this check to attribute error function?
    assert output_shape[split_dim] % num_splits == 0
    output_shape[split_dim] //= num_splits
    output_shape = tuple(output_shape)
    output_types = tuple(
        Tensor(dtype=x.dtype, shape=output_shape, device=x.device)
        for i in range(num_splits)
    )
    if op.op_type == "SplitUniformToTupleType":
        return TupleType(output_types)
    else:
        return output_types


def _softmax_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    return x


def _sqrt_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    return x


def _tanh_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    return x


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
    ("Concat", tuple(Tensor for _ in range(2))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(4))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(5))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 2))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 4))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 8))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 16))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 32))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 64))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 128))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 256))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 2))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 4))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 8))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 16))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 32))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 64))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 128))): _concat_prop_fn,
    ("Concat", tuple(Tensor for _ in range(3 * 256))): _concat_prop_fn,
    ("ConstantOfShape", (Tensor,)): _constant_of_shape_prop_fn,
    ("Div", (Tensor, Tensor)): _elementwise_tensor_op_prop_fn,
    ("Dropout", (Tensor, Tensor, type(Bool()))): _dropout_prop_fn,
    ("Expand", (Tensor, Tensor)): _expand_prop_fn,
    ("Gather", (Tensor, Tensor)): _gather_prop_fn,
    ("Gemm", (Tensor, Tensor, Tensor)): _gemm_prop_fn,
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
    ("MPIBroadcast", (Int32,)): _mpi_broadcast_prop_fn,
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
    ("Loss", (Tensor, Tensor, Int32)): _loss_prop_fn,
    ("LossGrad", (Tensor, Tensor, Int32)): _loss_grad_prop_fn,
    ("LayerNormalization", (Tensor, Tensor, Tensor)): _layer_norm_prop_fn,
    ("MatMul", (Tensor, Tensor)): _matmul_prop_fn,
    ("MatMulGrad", (Tensor, Tensor, Tensor)): _matmul_grad_prop_fn,
    ("Min", (Tensor, Tensor)): _min_prop_fn,
    ("Mul", (Tensor, Tensor)): _elementwise_tensor_op_prop_fn,
    ("ReduceMean", (Tensor,)): _reduce_mean_prop_fn,
    ("Relu", (Tensor,)): _relu_prop_fn,
    ("ReluGrad", (Tensor, Tensor)): _relu_grad_prop_fn,
    ("Select", (TupleType,)): _select_prop_fn,
    ("Send", (Tensor,)): _send_prop_fn,
    ("Send", (Int32,)): _send_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(4)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(8)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(16)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(32)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(64)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(128)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(256)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(512)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(1024)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(2048)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(4096)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(8192)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(16384)))): _sgd_prop_fn,
    ("SGDOptimizer", (tuple(Tensor for i in range(32768)))): _sgd_prop_fn,
    ("SplitUniform", (Tensor,)): _split_uniform_prop_fn,
    ("SplitUniformToTupleType", (Tensor,)): _split_uniform_prop_fn,
    ("Split", (Tensor,)): _split_prop_fn,
    ("Softmax", (Tensor,)): _softmax_prop_fn,
    ("Sqrt", (Tensor,)): _sqrt_prop_fn,
    ("Sub", (Tensor, Tensor)): _elementwise_tensor_op_prop_fn,
    ("Tanh", (Tensor,)): _tanh_prop_fn,
    ("Transpose", (Tensor,)): _transpose_prop_fn,
    ("Unsqueeze", (Tensor,)): _unsqueeze_prop_fn,
}
