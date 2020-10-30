class Op:
    def __init__(self, node, op_type):
        self._node = node
        self._op_type = op_type

    @property
    def node(self):
        return self._node

    @property
    def op_type(self):
        return self._op_type
