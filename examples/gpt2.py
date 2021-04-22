import argparse
from collections import defaultdict
import numpy as np
from transformers import GPT2Tokenizer
import torch

from dist_ir.executor import (
    CostModel,
    infer_types,
    PostTypeInferenceSimulator,
    Simulator,
    SequentialExecutor,
)
from dist_ir.importer import import_from_onnx
from dist_ir.ir import cpprint, Device, Topology, Value
from dist_ir.ir.type import Float32, Tensor


def to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x


def main(args):
    topology = Topology()
    d0 = topology.add_device("gpu")
    function, input_data = import_from_onnx(
        args.model_path, default_device=d0, parse_input_data=True
    )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(
        "Here is some text to encode Hello World", add_special_tokens=True
    )
    input_ids = torch.tensor([[tokens for _ in range(args.batch_size)]])
    input_ids = to_numpy(input_ids)

    inputs_with_shapes = [
        Value(
            function.inputs[0].name,
            Tensor(
                dtype=Float32(),
                shape=tuple(input_ids.shape),
                device=d0,
            ),
        )
    ]
    inputs_with_shapes += list(input_data.keys())
    input_data = [input_ids] + list(input_data.values())
    inputs = []
    for i in range(len(function.inputs)):
        if (
            i == 0
            or "weight" in function.inputs[i].name
            or "bias" in function.inputs[i].name
        ):
            inputs.append(inputs_with_shapes[i].type)
        else:
            assert inputs_with_shapes[i].type.shape == (1,)
            inputs.append(input_data[i])
    """
    function = infer_types_with_mixed_inputs(function, inputs)
    """
    ex = SequentialExecutor("numpy")
    function = ex.infer_types(function, input_data)
    simulator = PostTypeInferenceSimulator(CostModel(topology))
    simulation = simulator.interpret(function, (v.type for v in function.inputs))

    op_costs = defaultdict(list)
    for event in simulation.trace:
        op_costs[event["name"]].append(event["dur"])
    for op_type in op_costs:
        print(f"{op_type}: {np.median(op_costs[op_type]) * 1e6} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to ONNX model"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    main(args)
