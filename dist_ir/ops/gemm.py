from .op import Op
from ..graph.tensor import Tensor
from ..graph.value import Value


class GeMM(Op):
    def __init__(self, node, impl=None):
        super().__init__(node=node, op_type="gemm", impl=impl)

    def compute(self, v1: Value, v2: Value) -> Value:
        if not (isinstance(v1, Tensor) and isinstance(v2, Tensor)):
            raise ValueError(f"Unsupported input types {type(v1)} and {type(v2)}")

        if self._impl is None:
            raise RuntimeError("No implementation specified!")

        output_name = f"{self._node.name}/output"
        output_data = self._impl(v1.data, v2.data)
        ret = Tensor(name=output_name, data=output_data)
        return ret
