from .graph import Graph

import onnx


def import_from_onnx(onnx_model, backend):
    onnx_model = onnx.load(onnx_model)
    dist_ir_graph = Graph(backend=backend)

    inputs = {}
    output_src = {}

    for value in onnx_model.graph.value_info:
        print(f"Adding input {value.name} from graph.value_info")
        inputs[value.name] = dist_ir_graph.add_input(value.name)
    print()

    for value in onnx_model.graph.input:
        print(f"Adding input {value.name} from graph.input")
        inputs[value.name] = dist_ir_graph.add_input(value.name)
    print()

    for node in onnx_model.graph.node:
        per_node_inputs = []
        print(f"Getting inputs for node {node.name}...")
        for value in node.input:
            if value in inputs:
                print(f"Found input {value} in inputs")
            elif value in output_src:
                print(f"Found input {value} in output_src")
            else:
                print(f"---> Could not find input {value}!")
        print()
        # TODO: Set inputs
        dist_ir_graph.add_node(name=node.name, op_type=node.op_type)
        for output in node.output:
            output_src[output] = node
