from .graph import Graph

import onnx


def import_from_onnx(onnx_model, backend):
    onnx_model = onnx.load(onnx_model)
    dist_ir_graph = Graph(backend=backend)

    for tensor in onnx_model.graph.initializer:
        dist_ir_graph.add_input(tensor.name)

    for node in onnx_model.graph.node:
        # TODO: Set inputs
        dist_ir_graph.add_node(name=node.name, op_type=node.op_type)
