import numpy as np


def add(op, inputs):
    return np.add(inputs[0], inputs[1])


def allreduce(op, inputs):
    # TODO: Add attribute for reduction operator
    sum_ = np.sum(inputs[0], axis=0)
    return [sum_ for i in range(len(inputs[0]))]


def broadcast(op, inputs):
    return [inputs[0] for _ in range(len(op.attributes["devices"]))]


def concat(op, inputs):
    # assert len(inputs) == 1
    # dim = op.attributes["dim"]
    # return np.concatenate(inputs[0], axis=dim)
    dim = op.attributes["dim"]
    return np.concatenate(inputs, axis=dim)


def gather(op, inputs):
    dim = op.attributes["dim"]
    return np.concatenate(inputs[0], axis=dim)


def identity(op, inputs):
    return inputs[0]


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


def relu(op, inputs):
    return np.maximum(inputs[0], 0)


def select(op, inputs):
    dim = op.attributes["dim"]
    return inputs[0][dim]


def split(op, inputs):
    dim = op.attributes["dim"]
    if op.op_type == "Split":
        num_splits = op.attributes["num_splits"]
    elif op.op_type == "Scatter":
        num_splits = len(op.attributes["devices"])

    return np.split(inputs[0], num_splits, axis=dim)


def transpose(op, inputs):
    return inputs[0].T


NumPyRegister = {
    "Add": add,
    "Allreduce": allreduce,
    "Broadcast": broadcast,
    "Concat": concat,
    "Gather": gather,
    "Loss": loss,
    "LossGrad": loss_grad,
    "MatMul": matmul,
    "MatMulGrad": matmul_grad,
    "Relu": relu,
    "Scatter": split,
    "Select": select,
    "Send": identity,
    "Split": split,
    "Transpose": transpose,
}
