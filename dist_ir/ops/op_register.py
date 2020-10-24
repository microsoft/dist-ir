from .add import Add
from .matmul import MatMul

OpRegister = {
    'add': Add,
    'matmul': MatMul,
}
