from .backend_register import BackendRegister
from ..ir.type import Tensor


class SequentialExecutor:
    def __init__(self, backend):
        if backend not in BackendRegister:
            raise ValueError(f"Unknown backend {backend}")
        self._backend = backend

    def _resolve_inputs(self, inputs):
        """Converts the given inputs into the form expected by the specified backend."""
        resolved_inputs = []
        for input in inputs:
            # TODO: Support input types beyond Tensor
            if not isinstance(input.type, Tensor):
                raise ValueError(f"Invalid input type {input.type}")
            resolved_inputs.append(input.data)
        return resolved_inputs

    def _compute_node(self, node, inputs):
        """Executes the given node and returns its outputs."""
        op_type = node.op.op_type
        if op_type not in BackendRegister[self._backend]:
            raise NotImplementedError(
                f"No {self._backend} implementation found for op {op_type}"
            )
        impl = BackendRegister[self._backend][op_type]
        resolved_inputs = self._resolve_inputs(inputs)
        output_name = f"{node.name}/output"
        output_data = impl(*resolved_inputs)
        # TODO: Support output types beyond Tensor
        # TODO: Support multiple output values
        return Tensor(name=output_name, data=output_data)

    def compute(self, graph, input_data):
        """Executes the graph given the specified inputs and returns the final result.

        Args:
          graph: The graph to execute.
          input_data: A map from input tensor name to data represented in the
                      specified backend.

        Returns:
          A map from output tensor name to output tensor.
        """
        consumers = {}
        outputs = {}
        nodes = graph.get_nodes()

        # Execute ops in topological order.
        for node_name in nodes:
            node = nodes[node_name]
            inputs = []
            in_edges = node.get_in_edges()
            for in_edge in in_edges:
                if graph.is_node(in_edge):
                    input_node_name = in_edge
                    if input_node_name not in outputs:
                        raise RuntimeError(
                            f"Could not find node {input_node_name} as input for node {node.name}"
                        )
                    inputs.append(outputs[input_node_name])
                    consumers[input_node_name] -= 1
                elif graph.is_input(in_edge):
                    input_tensor_name = in_edge
                    if input_tensor_name not in input_data:
                        raise ValueError(
                            f"Could not find input {input_tensor_name} in input_data"
                        )
                    input_tensor = graph.get_input(input_tensor_name)
                    input_tensor.data = input_data[input_tensor_name]
                    inputs.append(input_tensor)
                else:
                    raise RuntimeError(f"Invalid in edge {in_edge}")
            res = self._compute_node(node, inputs)
            outputs[node.name] = res
            consumers[node.name] = len(node.get_out_edges())

            # Garbage collect any output tensors that have been fully consumed.
            to_free = []
            for in_edge in in_edges:
                if graph.is_node(in_edge):
                    input_node_name = in_edge
                    if consumers[input_node_name] == 0:
                        if len(graph.get_node(input_node_name).get_out_edges()) > 0:
                            to_free.append(input_node_name)
            for input_node_name in to_free:
                del outputs[input_node_name]
                del consumers[input_node_name]

        # Return the outputs.
        return outputs
