from ..ops.op import Op

class Node:
    def __init__(self, name, op_type):
        self._name = name
        self._in_edges = []
        self._out_edges = []
        self._op = Op(self, op_type)

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
