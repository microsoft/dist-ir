import numpy as np


def add(op, inputs):
    return np.add(inputs[0], inputs[1])


def allreduce(op, inputs):
    # TODO: Add attribute for reduction operator
    sum_ = np.sum(inputs[0], axis=0)
    return [sum_ for i in range(len(inputs[0]))]


def broadcast(op, inputs):
    return [inputs[0] for _ in range(len(op.attributes["devices"]))]


def cast(op, inputs):
    proto_dtype = op.attributes["to"]
    if proto_dtype == 0:
        raise ValueError("Undefined data type")
    elif proto_dtype == 1:
        return inputs[0].astype(np.float32)
    elif proto_dtype == 6:
        return inputs[0].astype(np.int32)
    elif proto_dtype == 7:
        return inputs[0].astype(np.int64)
    elif proto_dtype == 9:
        return inputs[0].as_type(np.bool_)
    else:
        raise NotImplementedError(f"Unsupported data type {proto_dtype}")


def concat(op, inputs):
    dim = op.attributes["dim"]
    return np.concatenate(inputs, axis=dim)


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
    return np.concatenate(inputs[0], axis=dim)


def identity(op, inputs):
    return inputs[0]


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
    return np.square(inputs[0] - inputs[1]) / N


def loss_grad(op, inputs):
    N = op.attributes["N"]
    return 2 * (inputs[0] - inputs[1]) / N


def matmul(op, inputs):
    return np.matmul(inputs[0], inputs[1])


def matmul_grad(op, inputs):
    return (np.dot(inputs[2], inputs[1].T), np.dot(inputs[0].T, inputs[2]))


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
    return inputs[0][dim]


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

    return np.split(inputs[0], num_splits, axis=dim)


def sub(op, inputs):
    return inputs[0] - inputs[1]


def transpose(op, inputs):
    perm = op.attributes["perm"]
    return np.transpose(inputs[0], perm)


def unsqueeze(op, inputs):
    x = inputs[0]
    axes = op.attributes["axes"]
    # TODO: Does this need to be in reverse order?
    for i in axes:
        x = np.expand_dims(x, axis=i)
    return x


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
