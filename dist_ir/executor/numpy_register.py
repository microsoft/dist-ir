import numpy as np


def add(op, inputs):
    return np.add(inputs[0], inputs[1])


def concat(op, inputs):
    dim = op.get_attribute("dim")
    return np.concatenate(inputs, axis=dim)


def identity(op, inputs):
    return inputs[0]


def loss(op, inputs):
    N = op.get_attribute("N")
    return np.square(inputs[0] - inputs[1]) / N


def loss_grad(op, inputs):
    N = op.get_attribute("N")
    return 2 * (inputs[0] - inputs[1]) / N


def matmul(op, inputs):
    return np.matmul(inputs[0], inputs[1])


def matmul_grad(op, inputs):
    return (np.dot(inputs[2], inputs[1].T), np.dot(inputs[0].T, inputs[2]))


def relu(op, inputs):
    return np.maximum(inputs[0], 0)


def select(op, inputs):
    dim = op.get_attribute("dim")
    return inputs[0][dim]


def split(op, inputs):
    dim = op.get_attribute("dim")
    num_splits = op.get_attribute("num_splits")

    return np.split(inputs[0], num_splits, axis=dim)


NumPyRegister = {
    "Add": add,
    "Concat": concat,
    "Loss": loss,
    "LossGrad": loss_grad,
    "MatMul": matmul,
    "MatMulGrad": matmul_grad,
    "Relu": relu,
    "Select": select,
    "Send": identity,
    "Split": split,
}
