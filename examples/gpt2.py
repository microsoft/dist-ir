import numpy as np

from dist_ir.executor import infer_types, SequentialExecutor
from dist_ir.importer import import_from_onnx
from dist_ir.ir import cpprint, Device, Value
from dist_ir.ir.type import Float, Tensor


def main():
    default_device = Device(0, "gpu")
    onnx_model_path = "/Users/keshavsanthanam/workspace/gpt2/model.onnx"
    function, input_data = import_from_onnx(
        onnx_model_path, default_device=default_device, parse_input_data=True
    )
    batch_size = 64
    sequence_length = 512
    third_dim = 128
    inputs_with_shapes = [
        Value(
            function.inputs[0].name,
            Tensor(
                dtype=Float(),
                shape=(batch_size, sequence_length, third_dim),
                device=default_device,
            ),
        )
    ]
    inputs_with_shapes += list(input_data.keys())
    input_data = tuple(np.random.normal(size=inp.type.shape) for inp in inputs_with_shapes)
    cpprint(function)
    ex = SequentialExecutor("numpy")
    result = ex.compute(function, input_data)
    #function = infer_types(function, inputs_with_shapes)


if __name__ == "__main__":
    main()
