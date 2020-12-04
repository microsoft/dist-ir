import torch


def add(op, inputs):
    return torch.add(inputs[0], inputs[1])


def matmul(op, inputs):
    return torch.matmul(inputs[0], inputs[1])


TorchRegister = {
    "Add": add,
    "MatMul": matmul,
}
