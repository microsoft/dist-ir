import numpy as np

from ..ir.type import Tensor


def _elementwise_numpy_op_prop_fn(op, x, y):
    if isinstance(x, Tensor) and isinstance(y, np.float32):
        return x
    elif isinstance(x, np.float32) and isinstance(y, Tensor):
        return y
    else:
        _raise_type_error(op, x, y)


def _gather_prop_fn(op, x, y):
    # TODO: Compute the new shape directly instead of using numpy
    if not (
        isinstance(x, Tensor)
        and x.shape is not None
        and (isinstance(y, np.ndarray) or isinstance(y, np.int64))
    ):
        _raise_type_error(op, x, y)
    if x.device is None:
        _raise_type_error(op, x, y)
    device = x.device
    temp = np.zeros(x.shape)
    if "axis" in op.attributes:
        axis = op.attributes["axis"]
    else:
        axis = 0
    new_shape = np.take(temp, y.astype(np.int64), axis=axis).shape
    return Tensor(dtype=x.dtype, shape=new_shape, device=device)


def _reshape_prop_fn(op, x, y):
    if not (isinstance(x, Tensor) and isinstance(y, np.ndarray)):
        _raise_type_error(op, x, y)
    y = y.tolist()
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
    if not isinstance(x, Tensor):
        _raise_type_error(op, x, y)
    return x


def _slice_prop_fn(op, x, starts, ends, axes, steps):
    if not (
        isinstance(x, Tensor)
        and isinstance(starts, np.ndarray)
        and isinstance(ends, np.ndarray)
        and isinstance(axes, np.ndarray)
        and (isinstance(steps, np.ndarray) or isinstance(steps, np.int64))
    ):
        _raise_type_error(op, x, starts, ends, axes, steps)
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
    return np.array(x.shape, dtype=np.int64)


MixedImplementations = {
    ("Add", (Tensor, np.float32)): _elementwise_numpy_op_prop_fn,
    ("Div", (Tensor, np.float32)): _elementwise_numpy_op_prop_fn,
    ("Gather", (Tensor, np.ndarray)): _gather_prop_fn,
    ("Gather", (Tensor, np.int64)): _gather_prop_fn,
    ("Mul", (Tensor, np.float32)): _elementwise_numpy_op_prop_fn,
    ("Reshape", (Tensor, np.ndarray)): _reshape_prop_fn,
    ("Pow", (Tensor, np.float32)): _pow_prop_fn,
    ("Slice", (Tensor, np.ndarray, np.ndarray, np.ndarray, np.int64)): _slice_prop_fn,
    ("Shape", (Tensor,)): _shape_prop_fn,
    ("Sub", (np.float32, Tensor)): _elementwise_numpy_op_prop_fn,
}
