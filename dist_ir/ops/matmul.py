from .op import Op
from .tensor import Tensor

class MatMul(Op):
    def __init__(self, impl=None):
        super().__init__(op_type='matmul', impl=impl)

    def compute(self, t1: Tensor, t2: Tensor) -> Tensor:
        if self._impl is None:
            raise RuntimeError('No implementation specified!')
        ret = Tensor(self._impl(t1.data, t2.data))
        return ret
