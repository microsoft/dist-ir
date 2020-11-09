from .module import Module
from .type import Tensor, Float
from .value import Value

import onnx


def import_from_onnx(onnx_model):
    # TODO: Remove prints?
    # TODO: Support types beyond Tensor
    onnx_model = onnx.load(onnx_model)
    dist_ir_module = Module()

    inputs = {}
    output_src = {}

    def add_input(value):
        # TODO lookup shape and dtype of input if exists
        v = dist_ir_module.add_input_value(value.name, Tensor(Float()))
        inputs[value.name] = v

    for value in onnx_model.graph.value_info:
        print(f"Adding input {value.name} from graph.value_info")
        add_input(value)
    print()

    for value in onnx_model.graph.input:
        print(f"Adding input {value.name} from graph.input")
        add_input(value)
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
                # TODO do something better here
                v = dist_ir_module.add_input_value(value, Tensor(Float()))
                inputs[value] = v
                per_node_inputs.append(v)
        print()
        op = dist_ir_module.add_op(node.op_type, node.name, per_node_inputs)
        for output in node.output:
            # TODO lookup shape and dtype of input if exists
            v = Value(output, Tensor(Float()))
            output_src[output] = v

    dist_ir_module.verify_ops_in_topological_order()
    return dist_ir_module
