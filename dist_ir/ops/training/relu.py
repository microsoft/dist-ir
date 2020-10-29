from ..op import Op


class ReLUGrad(Op):
    def __init__(self, node, impl=None):
        super().__init__(node=node, op_type="ReluGrad", impl=impl)
