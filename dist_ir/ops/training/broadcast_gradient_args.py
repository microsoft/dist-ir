from ..op import Op


class BroadcastGradientArgs(Op):
    def __init__(self, node, impl=None):
        super().__init__(node=node, op_type="BroadcastGradientArgs", impl=impl)
