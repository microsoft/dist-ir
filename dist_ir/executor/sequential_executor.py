class SequentialExecutor:
    def __init__(self):
        pass

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
        node_ids = graph.get_nodes_in_topological_order()

        # Execute ops in topological order.
        for node_id in node_ids:
            inputs = []
            node = graph.get_node(node_id)
            in_edges = node.get_in_edges()
            for in_edge in in_edges:
                if graph.is_node(in_edge):
                    input_node_id = in_edge
                    if input_node_id not in outputs:
                        raise RuntimeError(
                            f"Could not find node {input_node_id} as input for node {node_id}"
                        )
                    inputs.append(outputs[input_node_id])
                    consumers[input_node_id] -= 1
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
            res = node.op.compute(*inputs)
            outputs[node_id] = res
            consumers[node_id] = len(node.get_out_edges())

            # Garbage collect any output tensors that have been fully consumed.
            to_free = []
            for in_edge in in_edges:
                if graph.is_node(in_edge):
                    input_node_id = in_edge
                    if consumers[input_node_id] == 0:
                        if len(graph.get_node(input_node_id).get_out_edges()) > 0:
                            to_free.append(input_node_id)
            for input_node_id in to_free:
                del outputs[input_node_id]
                del consumers[input_node_id]

        # Return the outputs.
        return outputs