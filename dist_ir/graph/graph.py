from collections import OrderedDict

from .node import Node
from .tensor import Tensor


class Graph:
    def __init__(self, backend=None):
        self._nodes = OrderedDict()
        self._inputs = OrderedDict()
        self._op_counter = {}

    def get_nodes(self):
        """Returns all nodes in the graph."""
        return self._nodes

    def is_node(self, name):
        """Checks whether a node exists with the specified name."""
        return name in self._nodes

    def is_input(self, name):
        """Checks whether an input tensor exists with the specified name."""
        return name in self._inputs

    def get_node(self, name):
        """Returns the node with the specified name if it exists."""
        if name not in self._nodes:
            return None
        return self._nodes[name]

    def get_input(self, name):
        """Returns the input tensor with the specified name if it exists."""
        if name not in self._inputs:
            return None
        return self._inputs[name]

    def add_node(self, name, op_type, *inputs):
        """Adds a node to the graph.

        Args:
          op_type: The node's op type.
          inputs: The inputs for this node. These may either be other nodes or tensors.

        Returns:
          The newly created node.
        """
        if op_type not in self._op_counter:
            self._op_counter[op_type] = 0
        if name in self._nodes:
            raise ValueError(f"Node with name {name} already exists!")
        elif name is None or name == "":
            name = f"{op_type}/_{self._op_counter[op_type]}"
        node = Node(name, op_type)
        for in_edge in inputs:
            if isinstance(in_edge, Node):
                node.add_in_edge(in_edge.name)
                self._nodes[in_edge.name].add_out_edge(name)
            elif isinstance(in_edge, Tensor):
                node.add_in_edge(in_edge.name)
            else:
                raise ValueError(f"Invalid in edge type {type(in_edge)}")
        self._nodes[name] = node
        self._op_counter[op_type] += 1
        return node

    def add_input_tensor(self, name, data=None):
        """Adds an input tensor to the graph and returns the tensor."""
        tensor = Tensor(name, data)
        self._inputs[name] = tensor
        return tensor

    def _get_nodes_in_topological_order_helper(self, name, visited, order):
        visited.add(name)

        out_edges = self._nodes[name].get_out_edges()
        for out_edge in out_edges:
            output_name = out_edge
            if output_name not in visited:
                self._get_nodes_in_topological_order_helper(output_name, visited, order)

        order.append(name)

    def get_nodes_in_topological_order(self):
        """Return nodes in topological order."""
        visited = set()
        order = []
        for name in self._nodes:
            if name not in visited:
                self._get_nodes_in_topological_order_helper(name, visited, order)
        return order[::-1]

    def verify_nodes_in_topological_order(self):
        seen = set()
        for input in self._inputs:
            seen.add(input)

        for name, node in self._nodes.items():
            for in_edge in node.get_in_edges():
                if in_edge not in seen:
                    raise ValueError(f"Node are not in topological order: node {name} has unseen edge {in_edge}")
            seen.add(name)
