from .op_register import OpRegister
from .type import *
from .value import Value


class Op:
    def __init__(self, name, op_type, in_edges=None, attributes=None, submodules=None):
        if op_type not in OpRegister:
            raise ValueError(f"Invalid op type {op_type}")
        self._name = name
        self._op_type = op_type
        if in_edges is None:
            self._in_edges = []
        else:
            self._in_edges = in_edges
        self._out_edges = []
        # TODO: Handle non-tensor inputs
        input_shapes = tuple([in_edge.type.shape for in_edge in self._in_edges])
        output_shapes = OpRegister[op_type].infer_shapes(input_shapes)
        for i, output_shape in enumerate(output_shapes):
            output_name = f"{self._name}/{i}"
            output_value = Value(output_name, Tensor(Float(), output_shape))
            self._out_edges.append(output_value)
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
