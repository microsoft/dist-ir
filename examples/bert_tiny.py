import onnx
from pathlib import Path

import dist_ir
from dist_ir.importer import import_from_onnx

ONNX_MODEL_PATH = "onnx_models"


def main():
    onnx_model_path = Path(__file__).parent / ONNX_MODEL_PATH / "bert_tiny_bw.onnx"
    function = import_from_onnx(onnx_model_path)


def get_missing_op_types():
    onnx_model_path = Path(__file__).parent / ONNX_MODEL_PATH / "bert_tiny_bw.onnx"
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
