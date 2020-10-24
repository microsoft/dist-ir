class Op:
    def __init__(self, node, op_type, impl=None):
        self._node = node
        self._op_type = op_type
        self._impl = impl

    def bind_impl(self, impl):
        self._impl = impl

    @property
    def node(self):
        return self._node

    @property
    def op_type(self):
        return self._op_type
