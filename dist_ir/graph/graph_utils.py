from .graph import Graph

import onnx

def import_from_onnx(onnx_model, backend):
    onnx_model = onnx.load(onnx_model)
    dist_ir_graph = Graph(backend=backend)

    # Add weights as tensors.
    for tensor in onnx_model.graph.initializer:
        # TODO: Get data using ONNX numpy helper
        dist_ir_graph.add_tensor(tensor.name, data=None)

    for node in onnx_model.graph.node:
        # TODO: Set inputs
        dist_ir_graph.add_node(op_type=node.op_type)
