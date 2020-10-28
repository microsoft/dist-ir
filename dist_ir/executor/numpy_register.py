import numpy as np

NumPyRegister = {
    "Add": np.add,
    "MatMul": np.matmul,
    "Relu": lambda x: np.maximum(x, 0),
}
