from .op import Op
from .tensor import Tensor


class MatMul(Op):
    def __init__(self, node, impl=None):
        super().__init__(node=node, op_type="MatMul", impl=impl)

    def compute(self, t1: Tensor, t2: Tensor) -> Tensor:
        if self._impl is None:
            raise RuntimeError("No implementation specified!")
        output_name = f"{self._node.name}/output"
        output_data = self._impl(t1.data, t2.data)
        ret = Tensor(name=output_name, data=output_data)
        return ret
