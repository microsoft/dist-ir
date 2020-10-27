from collections import OrderedDict

from .node import Node
from ..executor.backend_register import BackendRegister
from ..ops.tensor import Tensor


class Graph:
    def __init__(self, backend=None):
        self._nodes = OrderedDict()
        self._inputs = OrderedDict()
        self._node_id_counter = {}
        if backend is not None and backend not in BackendRegister:
            raise ValueError(f"Unknown backend {backend}")
        else:
            self._backend = backend

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

    def add_node(self, op_type, *inputs):
        """Adds a node to the graph.

        Args:
          op_type: The node's op type.
          inputs: The inputs for this node. These may either be other nodes or tensors.

        Returns:
          The newly created node.
        """
        if op_type not in self._node_id_counter:
            self._node_id_counter[op_type] = 0
        node_id = f"{op_type}_{self._node_id_counter[op_type]}"
        self._node_id_counter[op_type] += 1
        node = Node(node_id, op_type)
        self.set_backend_for_node(node)
        for in_edge in inputs:
            if isinstance(in_edge, Node):
                node.add_in_edge(in_edge.node_id)
                self._nodes[in_edge.node_id].add_out_edge(node_id)
            elif isinstance(in_edge, Tensor):
                node.add_in_edge(in_edge.name)
            else:
                raise ValueError(f"Invalid in edge type {type(in_edge)}")
        self._nodes[node_id] = node
        return node

    def add_input(self, name, data=None):
        """Adds an input tensor to the graph and returns the tensor."""
        tensor = Tensor(name, data)
        self._inputs[name] = tensor
        return tensor

    def _get_nodes_in_topological_order_helper(self, node_id, visited, order):
        visited[node_id] = True

        out_edges = self._nodes[node_id].get_out_edges()
        for out_edge in out_edges:
            output_node_id = out_edge
            if not visited[output_node_id]:
                self._get_nodes_in_topological_order_helper(
                    output_node_id, visited, order
                )

        order.append(node_id)

    def get_nodes_in_topological_order(self):
        """Return nodes in topological order."""
        visited = {}
        for node_id in self._nodes:
            visited[node_id] = False
        order = []
        for node_id in self._nodes:
            if not visited[node_id]:
                self._get_nodes_in_topological_order_helper(node_id, visited, order)
        return order[::-1]

    def set_backend_for_node(self, node):
        """Sets the backend implementation for the given node."""
        if self._backend is not None and node.op is not None:
            op_type = node.op.op_type
            if op_type not in BackendRegister[self._backend]:
                raise NotImplementedError(
                    f"No {self._backend} implementation found for op {op_type}"
                )
            else:
                impl = BackendRegister[self._backend][op_type]
                node.op.bind_impl(impl)

    def set_backend(self, backend):
        """Sets the backend implementation for all nodes in the graph."""
        if backend not in BackendRegister:
            raise ValueError(f"Unknown backend {backend}")
        else:
            self._backend = backend
        for node in self._nodes.values():
            self.set_backend_for_node(node)
