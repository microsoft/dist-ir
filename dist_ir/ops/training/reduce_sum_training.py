from ..op import Op


class ReduceSumTraining(Op):
    def __init__(self, node, impl=None):
        super().__init__(node=node, op_type="ReduceSumTraining", impl=impl)
