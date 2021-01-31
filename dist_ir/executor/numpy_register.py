import numpy as np


def add(op, x, y):
    return np.add(x, y)


def allreduce(op, xs):
    # TODO: Add attribute for reduction operator
    sum_ = np.sum(xs, axis=0)
    return [sum_ for i in range(len(xs))]


def broadcast(op, x):
    return [x for _ in range(len(op.attributes["devices"]))]


def cast(op, x):
    proto_dtype = op.attributes["to"]
    dtype = {
        1: np.float32,
        6: np.int32,
        7: np.int64,
        9: np.bool,
    }[proto_dtype]
    return x.astype(dtype)


def concat(op, xs):
    # TODO make variadic
    # assert len(inputs) == 1
    # dim = op.attributes["dim"]
    # return np.concatenate(inputs[0], axis=dim)
    dim = op.attributes["dim"]
    return np.concatenate(xs, axis=dim)


def gather(op, xs):
    dim = op.attributes["dim"]
    return np.concatenate(xs, axis=dim)


def identity(op, x):
    return x


def loss(op, x, y):
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


def select(op, xs):
    dim = op.attributes["dim"]
    return xs[dim]


def slice(op, x, starts, ends, axes):
    # TODO handle the other cases, e.g. negative indices
    slices = {axis: slice(s, e) for (s, e, axis) in zip(starts, ends, axes)}
    slices = tuple(slices.get(d, slice(None)) for d in range(x.ndim))
    return x[slices]


def split(op, x):
    dim = op.attributes["dim"]
    if op.op_type == "Split":
        num_splits = op.attributes["num_splits"]
    elif op.op_type == "Scatter":
        num_splits = len(op.attributes["devices"])
    else:
        raise NotImplementedError(op.op_type)

    return np.split(x, num_splits, axis=dim)


def transpose(op, x):
    return x.T


NumPyRegister = {
    ("Add", (np.ndarray, np.ndarray)): add,
    ("Allreduce", (tuple,)): allreduce,
    ("Broadcast", (np.ndarray,)): broadcast,
    ("Cast", (np.ndarray,)): cast,
    ("Concat", (tuple,)): concat,
    ("Gather", (tuple,)): gather,
    ("Loss", (np.ndarray, np.ndarray)): loss,
    ("LossGrad", (np.ndarray, np.ndarray)): loss_grad,
    ("MatMul", (np.ndarray, np.ndarray)): matmul,
    ("MatMulGrad", (np.ndarray, np.ndarray, np.ndarray)): matmul_grad,
    ("Min", (np.ndarray, np.ndarray)): lambda op, x, y: np.minimum(x, y),
    ("Relu", (np.ndarray,)): relu,
    ("Scatter", (np.ndarray,)): split,
    ("Select", (np.ndarray,)): select,
    ("Send", (np.ndarray,)): identity,
    ("Split", (np.ndarray,)): split,
    ("Shape", (np.ndarray,)): lambda op, x: np.array(x.shape, dtype=np.int64),
    ("Slice", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): slice,
    ("Transpose", (np.ndarray,)): transpose,
}
