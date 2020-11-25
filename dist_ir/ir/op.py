from .op_register import OpRegister
from .type import *
from .value import Value


class Op:
    def __init__(
        self,
        name,
        op_type,
        device=None,
        in_edges=None,
        attributes=None,
        submodules=None,
        metadata=None,
        output_names=None,
    ):
        if op_type not in OpRegister:
            raise ValueError(f"Invalid op type {op_type}")
        self._name = name
        self._op_type = op_type
        self._device = device
        if in_edges is None:
            self._in_edges = []
        else:
            self._in_edges = in_edges
        if attributes is None:
            self._attributes = {}
        else:
            self._attributes = attributes
        if submodules is None:
            self._submodules = []
        else:
            self._submodules = submodules
        if metadata is None:
            self._metadata = {}
        else:
            self._metadata = metadata
        self._out_edges = []
        OpRegister[op_type].infer_types(self, output_names)

    def __str__(self):
        output = ""
        output += f"Name: {self._name}\n"
        output += f"Op type: {self._op_type}\n"
        output += "Inputs:\n"
        for in_edge in self._in_edges:
            output += "  " + str(in_edge) + "\n"
        output += "Outputs:\n"
        for out_edge in self._out_edges:
            output += "  " + str(out_edge) + "\n"
        if len(self._submodules) > 0:
            output += "Submodules:\n"
            for submodule in self._submodules:
                output += "\n".join(
                    ["  " + line for line in str(submodule).split("\n")]
                )
        return output

    def __repr__(self):
        return str(self)

    def add_in_edge(self, in_edge: Value):
        """Adds an input edge."""
        self._in_edges.append(in_edge)

    def add_out_edge(self, out_edge: Value):
        """Adds an output edge."""
        self._out_edges.append(out_edge)

    def get_in_edges(self):
        """Returns all input edges."""
        return self._in_edges

    def get_out_edges(self):
        """Returns all output edges."""
        return self._out_edges

    def get_attribute(self, attribute_name):
        """Returns the specified attributes, or throws error if it does not exist."""
        return self._attributes[attribute_name]

    def get_submodule(self, idx):
        """Returns the submodule at the specified index."""
        return self._submodules[idx]

    def get_metadata(self, key):
        """Returns the metadata with the given key."""
        return self._metadata[key]

    @property
    def name(self):
        return self._name

    @property
    def op_type(self):
        return self._op_type

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device