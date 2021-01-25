import onnx
from pathlib import Path

import dist_ir
from dist_ir.importer import import_from_onnx
from dist_ir.ir import cpprint
from dist_ir.executor import infer_types
from dist_ir.ir.type import Int64, Tensor

ONNX_MODEL_PATH = "onnx_models"

model = "bert_tiny_bw.onnx"


def main():
    batch_size = 64
    max_seq_len = 512
    max_pred_per_seq = 1
    onnx_model_path = Path(__file__).parent / ONNX_MODEL_PATH / model
    function, input_data = import_from_onnx(onnx_model_path, parse_input_data=False)
    cpprint(function)
    input_types = [
        Tensor(dtype=Int64(), shape=(batch_size, max_seq_len)),
        Tensor(dtype=Int64(), shape=(batch_size, max_seq_len)),
        Tensor(dtype=Int64(), shape=(batch_size, max_seq_len)),
        Tensor(dtype=Int64(), shape=(batch_size, max_pred_per_seq)),
        Tensor(dtype=Int64(), shape=(batch_size, max_pred_per_seq)),
        Tensor(dtype=Int64(), shape=(batch_size,)),
    ]
    input_types += [inp.type for inp in function.inputs[len(input_types) :]]
    function = infer_types(function, input_types)
    cpprint(function)


def get_missing_op_types():
    onnx_model_path = Path(__file__).parent / ONNX_MODEL_PATH / model
    onnx_model = onnx.load(onnx_model_path)
    missing_op_types = set()
    for node in onnx_model.graph.node:
        if node.op_type not in dist_ir.ir.op_register.OpRegister:
            missing_op_types.add(node.op_type)
    for op_type in missing_op_types:
        print(op_type)


if __name__ == "__main__":
    # get_missing_op_types()
    main()
