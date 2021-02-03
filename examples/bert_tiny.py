from collections import defaultdict
import numpy as np
import onnx
from pathlib import Path
from contextlib import redirect_stdout

import dist_ir
from dist_ir.importer import import_from_onnx, parse_tensor_from_file
from dist_ir.ir import cpprint, pformat
from dist_ir.executor import infer_types, SequentialExecutor
from dist_ir.ir.type import Int64, Tensor

ONNX_MODEL_PATH = "onnx_models"
BERT_TRAINING_DATA_PATH = "bert_training_data"
NUM_FEATURES_PER_SAMPLE = 6
model = "bert_tiny_bw.onnx"

np.random.seed(0)


def main():
    batch_size = 64
    training_mode = True
    learning_rate = 1.0e-2
    onnx_model_path = Path(__file__).parent / ONNX_MODEL_PATH / model
    function, input_data = import_from_onnx(onnx_model_path, parse_input_data=True)
    # TODO: Cache function and input data
    cpprint(function)

    samples = defaultdict(list)
    for i in range(batch_size):
        sample_input_paths = [
            Path(__file__).parent
            / BERT_TRAINING_DATA_PATH
            / f"sample{i}"
            / f"feature{j}.pb"
            for j in range(NUM_FEATURES_PER_SAMPLE)
        ]
        for j in range(NUM_FEATURES_PER_SAMPLE):
            samples[j].append(parse_tensor_from_file(sample_input_paths[j]))
    inputs = [np.stack(samples[j], axis=0) for j in range(NUM_FEATURES_PER_SAMPLE)]
    input_value_order = [0, 2, 4, 3, 5, 1]
    for i, inp in enumerate(inputs):
        j = input_value_order[i]
        input_data[function.inputs[j]] = inp
        print(function.inputs[j].name, inp.shape)
    input_types = [Tensor(dtype=Int64(), shape=tuple(inp.shape)) for inp in inputs]
    input_types += [inp.type for inp in function.inputs[len(input_types) :]]
    assert function.inputs[6].name == "Learning_Rate"
    assert function.inputs[7].name == "training_mode"
    input_data[function.inputs[6]] = learning_rate
    input_data[function.inputs[7]] = training_mode
    """
    function = infer_types(function, input_types, input_data)
    cpprint(function)
    """
    ex = SequentialExecutor("numpy")
    outputs = ex.compute(function, [input_data[i] for i in function.inputs])


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
