#import os, sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from collections import OrderedDict

from .node import Node
from ..executor.backend_register import BackendRegister

class Graph:
    def __init__(self, backend=None):
        self._nodes = OrderedDict()
        self._node_id_counter = 0
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
             inputs: The input nodes for this node.

           Returns:
             The newly created node.
        """
        node_id = self._node_id_counter
        self._node_id_counter += 1
        node = Node(node_id, op_type)
        self.set_backend_for_node(node)
        for input_node in inputs:
            node.add_in_edge(input_node.node_id)
            self._nodes[input_node.node_id].add_out_edge(node_id)
        self._nodes[node_id] = node
        return node

    def get_nodes_in_topological_order(self):
        """Return nodes in topological order."""
        # TODO
        return self._nodes

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
        nodes = self.get_nodes_in_topological_order()

        # Execute ops in topological order.
        for node_id, node in nodes.items():
            in_edges = node.get_in_edges()
            for input_node_id in in_edges:
                if input_node_id not in outputs:
                    raise RuntimeError(f'Could not find node {input_node_id} as input for node {node_id}')
                inputs.append(outputs[input_node_id])
                consumers[input_node_id] -= 1
            res = node.op.compute(*inputs)
            outputs[node_id] = res
            consumers[node_id] = len(node.get_out_edges())

            # Garbage collect any output tensors that have been fully consumed.
            to_free = []
            for input_node_id in in_edges:
                if consumers[input_node_id] == 0:
                    to_free.append(input_node_id)
            for input_node_id in to_free:
                del outputs[input_node_id]
                del consumers[input_node_id]

            inputs = []

        # Populate the output data to return.
        ret = []
        for node_id in outputs:
            ret.append(outputs[node_id])
        if len(ret) == 0:
            raise RuntimeError('Could not find any output!')
        elif len(ret) > 1:
            raise RuntimeError('Found more than 1 output tensor!')
        elif len(ret) == 1:
            return ret[0].data
