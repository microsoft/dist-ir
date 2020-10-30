from .graph import Graph

import onnx


def import_from_onnx(onnx_model, backend):
    # TODO: Remove prints?
    # TODO: Support types beyond Tensor
    onnx_model = onnx.load(onnx_model)
    dist_ir_graph = Graph(backend=backend)

    inputs = {}
    output_src = {}

    for value in onnx_model.graph.value_info:
        print(f"Adding input {value.name} from graph.value_info")
        inputs[value.name] = dist_ir_graph.add_input_tensor(value.name)
    print()

    for value in onnx_model.graph.input:
        print(f"Adding input {value.name} from graph.input")
        inputs[value.name] = dist_ir_graph.add_input_tensor(value.name)
    print()

    for node in onnx_model.graph.node:
        per_node_inputs = []
        print(f"Getting inputs for node {node.name}...")
        for value in node.input:
            if value in inputs:
                print(f"Found input {value} in inputs")
                per_node_inputs.append(inputs[value])
            elif value in output_src:
                print(f"Found input {value} in output_src")
                per_node_inputs.append(output_src[value])
            else:
                print(f"---> Could not find input {value}!")
                inputs[value] = dist_ir_graph.add_input_tensor(value)
                per_node_inputs.append(inputs[value])
        print()
        dist_ir_node = dist_ir_graph.add_node(node.name, node.op_type, *per_node_inputs)
        for output in node.output:
            output_src[output] = dist_ir_node 
