class Op:
    def __init__(self, op_type, impl=None):
        self._op_type = op_type
        self._impl = impl

    def bind_impl(self, impl):
        self._impl = impl

    @property
    def op_type(self):
        return self._op_type
