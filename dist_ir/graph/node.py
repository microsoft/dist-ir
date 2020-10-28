from ..ops.op_register import OpRegister


class Node:
    def __init__(self, name, op_type):
        self._name = name
        self._in_edges = []
        self._out_edges = []
        if op_type not in OpRegister:
            raise ValueError(f"Invalid op type {op_type}")
        self._op = OpRegister[op_type](self)

    def add_in_edge(self, in_edge):
        """Adds an input edge."""
        self._in_edges.append(in_edge)

    def add_out_edge(self, out_edge):
        """Adds an output edge."""
        self._out_edges.append(out_edge)

    def get_in_edges(self):
        """Returns all input edges."""
        return self._in_edges

    def get_out_edges(self):
        """Returns all output edges."""
        return self._out_edges

    @property
    def name(self):
        return self._name

    @property
    def op(self):
        return self._op
