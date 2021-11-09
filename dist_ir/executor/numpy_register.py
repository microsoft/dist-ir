# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import scipy


def _handle_negative_axis(axis, tensor_rank):
    return axis + tensor_rank if axis < 0 else axis


def _size_helper(shape, start, end):
    size = shape[start]
    for i in range(start + 1, end):
        size *= shape[i]
    return size


def add(op, x, y):
    return np.add(x, y)


def bias_fast_gelu_grad_dx(op, dy, x, b):
    kAlpha = np.sqrt(np.pi) * np.sqrt(0.5)
    kGamma = 0.044715
    kBeta = kGamma * kAlpha * 3.0
    input_shape = x.shape
    bias_shape = b.shape
    x_cube = np.power(x, 3)
    tanh_result = kAlpha * np.tanh(x + kGamma * x_cube)
    sech_sqr_result = 1 - (tanh_result * tanh_result)
    dx = dy * (
        0.5 * (tanh_result + sech_sqr_result * (kAlpha * x + kBeta + x_cube) + 1)
    )
    assert dx.shape == input_shape
    return dx


def cast(op, x):
    proto_dtype = op.attributes["to"]
    dtype = {
        1: np.float32,
        6: np.int32,
        7: np.int64,
        9: bool,
    }[proto_dtype]
    return x.astype(dtype)


def concat(op, *xs):
    # TODO make variadic
    dim = op.attributes["axis"]
    return np.concatenate(xs, axis=dim)


def constant(op):
    v = op.attributes["value"]
    return np.array(v)


def constant_of_shape(op, x):
    if "value" in op.attributes:
        value = op.attributes["value"]
    else:
        value = 0.0
    return np.full(shape=x.astype(np.int32), fill_value=value)


def div(op, x, y):
    return np.divide(x, y)


def dropout(op, x, ratio, training_mode):
    if training_mode:
        scale = 1.0 / (1.0 - ratio)
        mask = np.random.randint(0, 2, size=x.shape)
        x = scale * mask * x
        return x, mask
    else:
        return x


def dropout_grad(op, dy, mask, ratio, extra_input=None):
    # TODO: Figure out what extra input is
    if ratio == 0:
        return dy
    else:
        return mask * dy / (1.0 - ratio)


def expand(op, x, y):
    return x * np.ones(y)


def gather(op, x, y):
    if "axis" in op.attributes:
        axis = op.attributes["axis"]
        if axis != 0:
            raise NotImplementedError(f"Gather currently requires axis to be 0")
    else:
        axis = 0
    return x[y]


def identity(op, x):
    return x


def gemm(op, a, b, c):
    alpha = op.attributes["alpha"]
    beta = op.attributes["beta"]
    if "transA" in op.attributes and op.attributes["transA"]:
        a = a.T
    if "transB" in op.attributes and op.attributes["transB"]:
        b = b.T
    return np.matmul(alpha * a, beta * b) + c


def join(op, *xs):
    return tuple(xs)


def loss(op, x, y, n):
    return np.square(x - y) / n


def loss_grad(op, x, y, n):
    return 2 * (x - y) / n


def matmul(op, x, y):
    return np.matmul(x, y)


def matmul_grad(op, x, y, dz):
    return (np.dot(dz, y.T), np.dot(x.T, dz))
    # return (np.dot(x, dz), np.dot(y, dz))


def relu(op, x):
    return np.maximum(x, 0)


def relu_grad(op, x, dy):
    # TODO: fix
    dx = np.zeros(dy.shape)
    dx[dy > 0] = 1
    return dx


def mul(op, x, y):
    return x * y


def nonzero(op, x):
    return np.array(np.nonzero(x))


def reduce_mean(op, x):
    if "keepdims" in op.attributes:
        keepdims = op.attributes["keepdims"]
    else:
        keepdims = 1
    return np.mean(x, axis=tuple(op.attributes["axes"]), keepdims=keepdims)


def reduce_sum(op, x):
    if "keepdims" in op.attributes:
        keepdims = op.attributes["keepdims"]
    else:
        keepdims = 1
    return np.sum(x, axis=tuple(op.attributes["axes"]), keepdims=keepdims)


def relu(op, x):
    return np.maximum(x, 0)


def reshape(op, x, new_shape):
    new_shape = list(new_shape)
    for i in range(len(new_shape)):
        if new_shape[i] == 0:
            new_shape[i] = x.shape[i]
    return np.reshape(x, new_shape)


def select(op, xs):
    index = op.attributes["index"]
    return xs[index]


def sgd(op, *xs):
    weights = xs[: (len(xs) // 2)]
    gradients = xs[(len(xs) // 2) :]
    lr = op.attributes["lr"]
    updated_weights = []
    for w, dw in zip(weights, gradients):
        updated_weights.append(w - lr * dw)
    return tuple(updated_weights)


def shape(op, x):
    return np.array(x.shape, dtype=np.int64)


def slice_conc(op, x, starts, ends, axes, steps=None):
    # TODO handle the other cases, e.g. negative indices
    if steps is not None and isinstance(steps, np.ndarray) and len(steps.shape) == 0:
        steps = np.expand_dims(steps, 0)
    if steps is None:
        steps = [1] * len(starts)
    elif isinstance(steps, np.int64):
        steps = [steps] * len(starts)
    else:
        assert len(steps) == len(starts)
    slices = {
        axis: slice(s, e, step) for (s, e, axis, step) in zip(starts, ends, axes, steps)
    }
    slices = tuple(slices.get(d, slice(None)) for d in range(x.ndim))
    return x[slices]


def softmax(op, x):
    axis = op.attributes["axis"]
    return scipy.special.softmax(x, axis=axis)


def split_uniform(op, x):
    dim = op.attributes["axis"]
    num_splits = op.attributes["num_splits"]
    return tuple(y for y in np.split(x, num_splits, axis=dim))


def split(op, x):
    split = op.attributes["split"]
    sections = []
    n = 0
    for s in split[:-1]:
        sections.append(n + s)
        n += s
    axis = op.attributes["axis"]
    return tuple(np.split(x, sections, axis=axis))


def sub(op, x, y):
    return x - y


def sum_(op, *xs):
    return sum(xs)


def tanh(op, x):
    return np.tanh(x)


def transpose(op, x):
    perm = op.attributes["perm"]
    return np.transpose(x, perm)


def unsqueeze(op, x):
    axes = op.attributes["axes"]
    for i in axes[::-1]:
        x = np.expand_dims(x, axis=i)
    return x


NumPyRegister = {
    ("Add", (np.ndarray, np.ndarray)): add,
    ("Add", (np.ndarray, np.float32)): add,
    (
        "BiasFastGeluGrad_dX",
        (np.ndarray, np.ndarray, np.ndarray),
    ): bias_fast_gelu_grad_dx,
    ("Cast", (np.ndarray,)): cast,
    ("Cast", (np.int64,)): cast,
    ("Cast", (np.float64,)): cast,
    ("Concat", (tuple,)): concat,
    ("Concat", (np.int64, np.int64)): lambda op, *xs: np.array(xs),
    ("Concat", (np.int64, np.int64, np.int64)): lambda op, *xs: np.array(xs),
    ("Concat", tuple(np.ndarray for _ in range(2))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3))): concat,
    ("Concat", tuple(np.ndarray for _ in range(4))): concat,
    ("Concat", tuple(np.ndarray for _ in range(5))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 2))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 4))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 8))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 16))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 32))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 64))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 128))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 256))): concat,
    ("concat", tuple(np.ndarray for _ in range(3 * 2))): concat,
    ("concat", tuple(np.ndarray for _ in range(3 * 4))): concat,
    ("concat", tuple(np.ndarray for _ in range(3 * 8))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 16))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 32))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 64))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 128))): concat,
    ("Concat", tuple(np.ndarray for _ in range(3 * 256))): concat,
    ("Constant", ()): constant,
    ("ConstantOfShape", (np.ndarray,)): constant_of_shape,
    ("Div", (np.ndarray, np.ndarray)): div,
    ("Div", (np.ndarray, np.float32)): div,
    ("Div", (np.int64, np.int64)): div,
    ("Dropout", (np.ndarray, np.ndarray, bool)): dropout,
    ("DropoutGrad", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): dropout_grad,
    ("Expand", (np.ndarray, np.ndarray)): expand,
    ("Gather", (np.ndarray, np.ndarray)): gather,
    ("Gather", (np.ndarray, np.int64)): gather,
    ("Gemm", (np.ndarray, np.ndarray, np.ndarray)): gemm,
    ("Identity", (np.ndarray,)): identity,
    ("Join", (np.ndarray, np.ndarray)): join,
    ("Join", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): join,
    ("Loss", (np.ndarray, np.ndarray, int)): loss,
    ("LossGrad", (np.ndarray, np.ndarray, int)): loss_grad,
    ("MatMul", (np.ndarray, np.ndarray)): matmul,
    ("MatMulGrad", (np.ndarray, np.ndarray, np.ndarray)): matmul_grad,
    ("Min", (np.ndarray, np.ndarray)): lambda op, x, y: np.minimum(x, y),
    ("Mul", (np.ndarray, np.ndarray)): mul,
    ("Mul", (np.ndarray, np.float32)): mul,
    ("Mul", (np.int64, np.int64)): mul,
    ("NonZero", (np.ndarray,)): nonzero,
    ("Pow", (np.ndarray, np.float32)): lambda op, x, y: pow(x, y),
    ("ReduceMean", (np.ndarray,)): reduce_mean,
    ("ReduceSum", (np.ndarray,)): reduce_sum,
    ("Relu", (np.ndarray,)): relu,
    ("ReluGrad", (np.ndarray, np.ndarray)): relu_grad,
    ("Reshape", (np.ndarray, np.ndarray)): reshape,
    ("Select", (tuple,)): select,
    ("Select", (np.ndarray,)): select,
    ("SGDOptimizer", tuple(np.ndarray for i in range(2))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(4))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(8))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(12))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(16))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(24))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(32))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(48))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(64))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(96))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(128))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(192))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(256))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(512))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(1024))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(2048))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(4096))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(8192))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(16384))): sgd,
    ("SGDOptimizer", tuple(np.ndarray for i in range(32768))): sgd,
    ("Shape", (np.ndarray,)): shape,
    ("Slice", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): slice_conc,
    ("Slice", (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int64)): slice_conc,
    ("SplitUniform", (np.ndarray,)): split_uniform,
    ("SplitUniformToTupleType", (np.ndarray,)): split_uniform,
    ("Split", (np.ndarray,)): split,
    ("Softmax", (np.ndarray,)): softmax,
    ("Sqrt", (np.ndarray,)): lambda op, x: np.sqrt(x),
    ("Squeeze", (np.ndarray,)): lambda op, x: np.squeeze(x),
    ("Sub", (np.ndarray, np.ndarray)): sub,
    ("Sub", (np.int64, np.int64)): sub,
    ("Sub", (np.float32, np.ndarray)): sub,
    ("Sum", (np.ndarray, np.ndarray)): sum_,
    ("Sum", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): sum_,
    ("Tanh", (np.ndarray,)): tanh,
    ("Transpose", (np.ndarray,)): transpose,
    ("Unsqueeze", (np.int64,)): unsqueeze,
    ("Unsqueeze", (np.ndarray,)): unsqueeze,
}
