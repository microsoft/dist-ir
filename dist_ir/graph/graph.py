#import os, sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from collections import OrderedDict

from .node import Node
from ..executor.backend_register import BackendRegister
from ..ops.tensor import Tensor

class Graph:
    def __init__(self, backend=None):
        self._nodes = OrderedDict()
        self._tensors = OrderedDict()
        self._node_id_counter = {}
        if backend is not None and backend not in BackendRegister:
            raise ValueError(f'Unknown backend {backend}')
        else:
            self._backend = backend

    def get_nodes(self):
        """Returns all nodes in the graph."""
        return self._nodes

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
        node_id = f'{op_type}_{self._node_id_counter[op_type]}'
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
                raise ValueError(f'Invalid in edge type {type(in_edge)}')
        self._nodes[node_id] = node
        return node

    def add_tensor(self, name, data):
        """Adds a new tensor to the graph and returns the tensor."""
        # TODO: Verify that the data type matches the backend
        tensor = Tensor(name=name, data=data)
        self._tensors[name] = tensor
        return tensor

    def _get_nodes_in_topological_order_helper(self, node_id, visited, order):
        visited[node_id] = True

        out_edges = self._nodes[node_id].get_out_edges()
        for out_edge in out_edges:
            output_node_id = out_edge
            if not visited[output_node_id]:
                self._get_nodes_in_topological_order_helper(output_node_id, visited, order)

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
                raise ValueError(f'No {self._backend} implementation found for op {op_type}')
            else:
                impl = BackendRegister[self._backend][op_type]
                node.op.bind_impl(impl)

    def set_backend(self, backend):
        """Sets the backend implementation for all nodes in the graph."""
        if backend not in BackendRegister:
            raise ValueError(f'Unknown backend {backend}')
        else:
            self._backend = backend
        for node in self._nodes.values():
            self.set_backend_for_node(node)

    def compute(self, *inputs):
        """Executes the graph given the specified inputs and returns the final result."""
        consumers = {}
        outputs = {}
        inputs = list(inputs)
        node_ids = self.get_nodes_in_topological_order()

        # Execute ops in topological order.
        for node_id in node_ids:
            node = self._nodes[node_id]
            in_edges = node.get_in_edges()
            for in_edge in in_edges:
                if in_edge in self._nodes:
                    input_node_id = in_edge
                    if input_node_id not in outputs:
                        raise RuntimeError(
                            f'Could not find node {input_node_id} as input for node {node_id}')
                    inputs.append(outputs[input_node_id])
                    consumers[input_node_id] -= 1
                elif in_edge in self._tensors:
                    input_tensor_name = in_edge
                    inputs.append(self._tensors[input_tensor_name])
                else:
                    raise RuntimeError(f'Invalid in edge {in_edge}')
            res = node.op.compute(*inputs)
            outputs[node_id] = res
            consumers[node_id] = len(node.get_out_edges())

            # Garbage collect any output tensors that have been fully consumed.
            to_free = []
            for in_edge in in_edges:
                if in_edge in self._nodes:
                    input_node_id = in_edge
                    if consumers[input_node_id] == 0:
                        if len(self._nodes[input_node_id].get_out_edges()) > 0:
                            to_free.append(input_node_id)
            for input_node_id in to_free:
                del outputs[input_node_id]
                del consumers[input_node_id]
            inputs = []

        # Return the outputs.
        return outputs
