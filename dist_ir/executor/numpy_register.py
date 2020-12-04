import numpy as np


def add(op, inputs):
    return np.add(inputs[0], inputs[1])


def loss(op, inputs):
    return np.square(inputs[0] - inputs[1]) / inputs[0].shape[0]


def loss_grad(op, inputs):
    return 2 * (inputs[0] - inputs[1]) / inputs[0].shape[0]


def matmul(op, inputs):
    return np.matmul(inputs[0], inputs[1])


def matmul_grad(op, inputs):
    return (np.dot(inputs[2], inputs[1].T), np.dot(inputs[0].T, inputs[2]))


def relu(op, inputs):
    return np.maximum(inputs[0], 0)


NumPyRegister = {
    "Add": add,
    "Loss": loss,
    "LossGrad": loss_grad,
    "MatMul": matmul,
    "MatMulGrad": matmul_grad,
    "Relu": relu,
}
