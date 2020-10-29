from .op import Op
from ..graph.tensor import Tensor


class SoftmaxCrossEntropy(Op):
    def __init__(self, node, impl=None):
        super().__init__(node=node, op_type="SoftmaxCrossEntropy", impl=impl)
