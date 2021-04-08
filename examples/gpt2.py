import numpy as np
from transformers import GPT2Tokenizer
import torch

from dist_ir.executor import infer_types, SequentialExecutor
from dist_ir.importer import import_from_onnx
from dist_ir.ir import cpprint, Device, Value
from dist_ir.ir.type import Float, Tensor

def to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x

def main():
    default_device = Device(0, "gpu")
    #onnx_model_path = "/Users/keshavsanthanam/workspace/gpt2/model.onnx"
    onnx_model_path = "/lfs/1/keshav2/gpt2/model.onnx"
    function, input_data = import_from_onnx(
        onnx_model_path, default_device=default_device, parse_input_data=True
    )
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids_1 = torch.tensor([[tokenizer.encode("Here is some text to encode Hello World", add_special_tokens=True)]])
    input_ids_1 = to_numpy(input_ids_1)
    
    inputs_with_shapes = [
        Value(
            function.inputs[0].name,
            Tensor(
                dtype=Float(),
                shape=tuple(input_ids_1.shape),
                device=default_device,
            ),
        )
    ]
    inputs_with_shapes += list(input_data.keys())
    input_data = [input_ids_1] + list(input_data.values())
    cpprint(function)
    ex = SequentialExecutor("numpy")
    result = ex.compute(function, input_data)
    #function = infer_types(function, inputs_with_shapes)


if __name__ == "__main__":
    main()
