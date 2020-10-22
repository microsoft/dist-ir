"""
Create a simple ONNX graph consisting of a single gemm from PyTorch
"""

import torch.nn as nn
import torch.onnx

N, D_in, D_out = 64, 100, 10

model = torch.nn.Linear(D_in, D_out)

# Test model:
x = torch.randn(N, D_in)
y_pred = model(x)
print(y_pred.shape)

# Export model as onnx file:
torch.onnx.export(
    model,
    x,
    "gemm.onnx",
    export_params=False,
    input_names=["x"],
    output_names=["preds"],
    dynamic_axes={"x": {0: "batch_size"}, "preds": {0: "batch_size"}},
)

# Here's how to create the same graph using the ONNX API:
def create_gemm_onnx():
    import onnx
    from onnx import helper
    from onnx import AttributeProto, TensorProto, GraphProto

    N, D_in, D_out = 64, 100, 10

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, D_in])
    weight = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [D_out, D_in])
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [D_out])
    preds = helper.make_tensor_value_info("preds", TensorProto.FLOAT, [N, D_out])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        "Gemm",  # Op type
        ["x", "weight", "bias"],  # inputs
        ["preds"],  # outputs
        name="Gemm_0",
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "data-parallel-model",
        [x, weight, bias],
        [preds],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name="dp_transform.py")

    with open("out.txt", "w") as fout:
        fout.write(str(model_def))