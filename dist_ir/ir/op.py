from .value import Value


class Op:
    def __init__(self, name, op_type, in_edges=None, attributes=None, submodules=None):
        self._name = name
        self._op_type = op_type
        if in_edges is None:
            self._in_edges = []
        else:
            self._in_edges = in_edges
        self._out_edges = []
        if attributes is None:
            self._attributes = {}
        else:
            self._attributes = attributes
        if submodules is None:
            self._submodules = []
        else:
            self._submodules = submodules

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

    def reset_out_edges(self):
        """Clears any existing output edges."""
        self._out_edges = []

    @property
    def name(self):
        return self._name

    @property
    def op_type(self):
        return self._op_type
