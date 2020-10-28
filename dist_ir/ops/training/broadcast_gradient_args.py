from ..op import Op
from ..tensor import Tensor


class BroadcastGradientArgs(Op):
    def __init__(self, node, impl=None):
        super().__init__(node=node, op_type="BroadcastGradientArgs", impl=impl)
