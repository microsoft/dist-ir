import numpy as np

NumPyRegister = {
    "Add": np.add,
    "Loss": lambda a, b: np.square(a - b) / a.shape[0],
    "LossGrad": lambda a, b: 2 * (a - b) / a.shape[0],
    "MatMul": np.matmul,
    "MatMulGrad": lambda a, b, c: (np.dot(c, b.T), np.dot(a.T, c)),
    "Relu": lambda x: np.maximum(x, 0),
}
