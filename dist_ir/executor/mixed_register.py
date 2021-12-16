# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This file defines a MixedRegister which contains op implementations that accept
combinations of concrete and abstract input types. This is necessary for certain
ops that require concrete arguments to perform type inference over abstract inputs.
For example, Reshape requires a concrete shape input to determine the output shape.
"""

import numpy as np

from .concrete_value import ConcreteValue
from ..ir.type import Tensor


def _raise_type_error(op, *args):
    raise ValueError(f"Type error: op\n{op}\nwas given arguments\n{tuple(args)}")


def _elementwise_numpy_op_prop_fn(op, x, y):
    if (
        isinstance(x, Tensor)
        and isinstance(y, ConcreteValue)
        and (y.val.dtype == np.float32 or y.val.dtype == np.float64)
    ):
        return x
    elif (
        isinstance(x, ConcreteValue)
        and isinstance(y, Tensor)
        and (x.val.dtype == np.float32 or x.val.dtype == np.float64)
    ):
        return y
    else:
        _raise_type_error(op, x, y)


def _gather_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, ConcreteValue)
        and x.shape is not None
        and y.val is not None
        and x.device == y.device
        and isinstance(y.val, np.ndarray)
    ):
        _raise_type_error(op, x, y)
    if x.device is None:
        _raise_type_error(op, x, y)
    device = x.device
    if "axis" in op.attributes:
        axis = op.attributes["axis"]
    else:
        axis = 0
    if isinstance(y.val, np.ndarray) and axis == 0:
        # Manually compute the new shape for the common case
        new_shape = y.val.shape + x.shape[1:]
    else:
        # Use the NumPy implementation in the general case
        temp = np.zeros(x.shape)
        new_shape = np.take(temp, y.val.astype(np.int64), axis=axis).shape
    return Tensor(dtype=x.dtype, shape=new_shape, device=device)


def _reshape_prop_fn(op, x, y):
    if not (
        isinstance(x, Tensor)
        and isinstance(y, ConcreteValue)
        and isinstance(y.val, np.ndarray)
    ):
        _raise_type_error(op, x, y)
    y = y.val.tolist()
    if y.count(-1) > 1:
        _raise_type_error(op, x, y)
    new_shape = []
    for dim in y:
        if dim != -1:
            new_shape.append(dim)
        else:
            new_shape.append(int(np.prod(x.shape) / np.prod(y) * -1))
    return Tensor(shape=tuple(new_shape), dtype=x.dtype, device=x.device)


def _pow_prop_fn(op, x, y):
    if not (isinstance(x, Tensor) and isinstance(y, ConcreteValue)):
        _raise_type_error(op, x, y)
    return x


def _slice_prop_fn(op, x, starts, ends, axes, steps):
    if not (
        isinstance(x, Tensor)
        and isinstance(starts, ConcreteValue)
        and isinstance(ends, ConcreteValue)
        and isinstance(axes, ConcreteValue)
        and isinstance(steps, ConcreteValue)
        and isinstance(starts.val, np.ndarray)
        and isinstance(ends.val, np.ndarray)
        and isinstance(axes.val, np.ndarray)
        and (isinstance(steps.val, np.ndarray) or isinstance(steps.val, np.int64))
        and x.device == starts.device
        and x.device == ends.device
        and x.device == axes.device
        and x.device == steps.device
    ):
        _raise_type_error(op, x, starts, ends, axes, steps)
    starts = starts.val
    ends = ends.val
    axes = axes.val
    steps = steps.val

    if len(steps.shape) == 0:
        steps = np.expand_dims(steps, 0)

    # TODO handle the other cases, e.g. negative indices
    assert -1 not in starts.tolist()
    assert -1 not in ends.tolist()
    assert -1 not in axes.tolist()
    if steps is None:
        steps = [1] * len(starts)
    elif isinstance(steps, np.int64):
        steps = [steps] * len(starts)
    else:
        assert len(steps) == len(starts)
    slices = {
        axis: slice(s, e, step) for (s, e, axis, step) in zip(starts, ends, axes, steps)
    }
    slices = tuple(slices.get(d, slice(None)) for d in range(len(x.shape)))
    new_shape = []
    for i, slice_ in enumerate(slices):
        start = slice_.start
        stop = slice_.stop
        step = slice_.step
        if start is None:
            start = 0
        if stop is None:
            stop = x.shape[i]
        if step is None:
            step = 1
        new_shape.append(int(np.ceil((stop - start) / step)))
    return Tensor(shape=tuple(new_shape), dtype=x.dtype, device=x.device)


def _shape_prop_fn(op, x):
    if not isinstance(x, Tensor):
        _raise_type_error(op, x)
    return ConcreteValue(np.array(x.shape, dtype=np.int64), x.device)


MixedRegister = {
    ("Add", (Tensor, ConcreteValue)): _elementwise_numpy_op_prop_fn,
    ("Div", (Tensor, ConcreteValue)): _elementwise_numpy_op_prop_fn,
    ("Gather", (Tensor, ConcreteValue)): _gather_prop_fn,
    ("Mul", (Tensor, ConcreteValue)): _elementwise_numpy_op_prop_fn,
    ("Reshape", (Tensor, ConcreteValue)): _reshape_prop_fn,
    ("Pow", (Tensor, ConcreteValue)): _pow_prop_fn,
    (
        "Slice",
        (Tensor, ConcreteValue, ConcreteValue, ConcreteValue, ConcreteValue),
    ): _slice_prop_fn,
    ("Shape", (Tensor,)): _shape_prop_fn,
    ("Sub", (ConcreteValue, Tensor)): _elementwise_numpy_op_prop_fn,
}
