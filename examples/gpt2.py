import argparse
import numpy as np
from transformers import GPT2Tokenizer
import torch

from dist_ir.executor import infer_types, SequentialExecutor
from dist_ir.importer import import_from_onnx
from dist_ir.ir import cpprint, Device, Value
from dist_ir.ir.type import Float32, Tensor


def to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x


def main(args):
    default_device = Device(0, "gpu")
    # onnx_model_path = "/Users/keshavsanthanam/workspace/gpt2/model.onnx"
    function, input_data = import_from_onnx(
        args.model_path, default_device=default_device, parse_input_data=True
    )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids_1 = torch.tensor(
        [
            [
                tokenizer.encode(
                    "Here is some text to encode Hello World", add_special_tokens=True
                )
            ]
        ]
    )
    input_ids_1 = to_numpy(input_ids_1)

    inputs_with_shapes = [
        Value(
            function.inputs[0].name,
            Tensor(
                dtype=Float32(),
                shape=tuple(input_ids_1.shape),
                device=default_device,
            ),
        )
    ]
    inputs_with_shapes += list(input_data.keys())
    input_data = [input_ids_1] + list(input_data.values())
    ex = SequentialExecutor("numpy")
    # result = ex.compute(function, input_data)
    function = ex.infer_types(function, input_data)
    cpprint(function)
    # function = infer_types(function, inputs_with_shapes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to ONNX model"
    )
    args = parser.parse_args()
    main(args)
