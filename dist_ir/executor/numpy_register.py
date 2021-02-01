import numpy as np


def add(op, x, y):
    return np.add(x, y)


def allreduce(op, xs):
    # TODO: Add attribute for reduction operator
    sum_ = np.sum(xs, axis=0)
    return tuple(sum_ for i in range(len(xs)))


def broadcast(op, x):
    return tuple(x for _ in range(len(op.attributes["devices"])))


def cast(op, x):
    proto_dtype = op.attributes["to"]
    dtype = {
        1: np.float32,
        6: np.int32,
        7: np.int64,
        9: np.bool,
    }[proto_dtype]
    return x.astype(dtype)


def concat2(op, x, y):
    dim = op.attributes["dim"]
    return np.concatenate((x, y), axis=dim)


def concat(op, xs):
    # TODO make variadic
    # assert len(inputs) == 1
    # dim = op.attributes["dim"]
    # return np.concatenate(inputs[0], axis=dim)
    dim = op.attributes["dim"]
    return np.concatenate(xs, axis=dim)


def div(op, inputs):
    return inputs[0] / inputs[1]


def dropout(op, inputs):
    x, ratio, training_mode = inputs
    if training_mode:
        scale = 1.0 / (1.0 - ratio)
        mask = np.random.randint(0, 2, size=x.shape)
        x = scale * mask * x
        assert x.shape == inputs[0].shape
        return x, mask
    else:
        return x


def expand(op, inputs):
    return inputs[0] * np.ones(inputs[1])


def fast_gelu(op, inputs):
    # https://github.com/hendrycks/GELUs
    x = inputs[0]
    return 1.0 / (1.0 + np.exp(-1.702 * x))


def gather(op, inputs):
    axis = op.attributes["axis"]
    return np.take(inputs[0], inputs[1].astype(np.int64), axis=axis)


def mpi_gather(op, inputs):
    dim = op.attributes["dim"]
    return np.concatenate(xs, axis=dim)


def identity(op, x):
    return x


def layer_norm(op, inputs):
    eps = 1e-5
    x, scale, beta = inputs
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / (pow(std, 2) + eps) * scale + beta
    assert x.shape == inputs[0].shape
    return x, mean, std


def loss(op, inputs):
    N = op.attributes["N"]
    return np.square(x - y) / N


def loss_grad(op, x, y):
    N = op.attributes["N"]
    return 2 * (x - y) / N


def matmul(op, x, y):
    return np.matmul(x, y)


def matmul_grad(op, x, y, dz):
    return (np.dot(dz, y.T), np.dot(x.T, dz))


def relu(op, x):
    return np.maximum(x, 0)


def min_(op, inputs):
    return np.minimum(*inputs)


def mul(op, inputs):
    return inputs[0] * inputs[1]


def relu(op, inputs):
    return np.maximum(inputs[0], 0)


def reshape(op, inputs):
    new_shape = np.copy(inputs[1])
    for i in range(len(new_shape)):
        if new_shape[i] == 0:
            new_shape[i] = inputs[0].shape[i]
    return np.reshape(inputs[0], new_shape)


def select(op, inputs):
    dim = op.attributes["dim"]
    return xs[dim]


def slice_conc(op, x, starts, ends, axes):
    # TODO handle the other cases, e.g. negative indices
    slices = {axis: slice(s, e) for (s, e, axis) in zip(starts, ends, axes)}
    slices = tuple(slices.get(d, slice(None)) for d in range(x.ndim))
    return x[slices]


def shape(op, inputs):
    return np.array(inputs[0].shape)


def slice_(op, inputs):
    x, starts, ends, axes = inputs
    slices = {axis: slice(s, e) for (s, e, axis) in zip(starts, ends, axes)}
    slices = tuple(slices.get(d, slice(None)) for d in range(x.ndim))
    return x[slices]


def softmax(op, inputs):
    axis = op.attributes["axis"]
    exp = np.exp(inputs[0])
    return exp / np.sum(exp, axis=axis, keepdims=True)


def split(op, inputs):
    dim = op.attributes["dim"]
    if op.op_type == "Split":
        num_splits = op.attributes["num_splits"]
    elif op.op_type == "Scatter":
        num_splits = len(op.attributes["devices"])
    else:
        raise NotImplementedError(op.op_type)

    return tuple(y for y in np.split(x, num_splits, axis=dim))


def transpose(op, x):
    perm = op.attributes["perm"]
    return np.transpose(x, perm)


def sub(op, inputs):
    return inputs[0] - inputs[1]


def unsqueeze(op, inputs):
    x = inputs[0]
    axes = op.attributes["axes"]
    # TODO: Does this need to be in reverse order?
    for i in axes:
        x = np.expand_dims(x, axis=i)
    return x


NumPyRegister = {
    ("Add", (np.ndarray, np.ndarray)): add,
    ("Allreduce", (tuple,)): allreduce,
    ("Broadcast", (np.ndarray,)): broadcast,
    ("Cast", (np.ndarray,)): cast,
    ("Concat", (tuple,)): concat,
    ("Concat", (np.ndarray, np.ndarray)): concat2,
    ("Gather", (tuple,)): gather,
    ("Loss", (np.ndarray, np.ndarray)): loss,
    ("LossGrad", (np.ndarray, np.ndarray)): loss_grad,
    ("MatMul", (np.ndarray, np.ndarray)): matmul,
    ("MatMulGrad", (np.ndarray, np.ndarray, np.ndarray)): matmul_grad,
    ("Min", (np.ndarray, np.ndarray)): lambda op, x, y: np.minimum(x, y),
    ("Relu", (np.ndarray,)): relu,
    ("Scatter", (np.ndarray,)): split,
    ("Select", (tuple,)): select,
    ("Send", (np.ndarray,)): identity,
    ("Split", (np.ndarray,)): split,
    ("Shape", (np.ndarray,)): lambda op, x: np.array(x.shape, dtype=np.int64),
    ("Slice", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): slice_conc,
    ("Transpose", (np.ndarray,)): transpose,
}

"""
NumPyRegister = {
    "Add": add,
    "Allreduce": allreduce,
    "Broadcast": broadcast,
    "Cast": cast,
    "Concat": concat,
    "Div": div,
    "Dropout": dropout,
    "Expand": expand,
    "FastGelu": fast_gelu,
    "Gather": gather,
    "Identity": identity,
    "LayerNormalization": layer_norm,
    "Loss": loss,
    "LossGrad": loss_grad,
    "MatMul": matmul,
    "MatMulGrad": matmul_grad,
    "Min": min_,
    "MPIGather": mpi_gather,
    "Mul": mul,
    "Relu": relu,
    "Reshape": reshape,
    "Scatter": split,
    "Select": select,
    "Send": identity,
    "Shape": shape,
    "Slice": slice_,
    "Softmax": softmax,
    "Split": split,
    "Sub": sub,
    "Transpose": transpose,
    "Unsqueeze": unsqueeze,
}
"""
