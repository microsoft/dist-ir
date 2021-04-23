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
from dist_ir.transforms import gpt2_dhp_transform

NETWORK_BANDWIDTH_Gbps = 200


def to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x


def main(args):
    topology = Topology()
    world_size = args.dp_degree * args.hp_degree * args.pp_degree
    d0 = topology.add_device("gpu")
    for i in range(world_size):
        topology.add_device("gpu")
        for j in range(i):
            topology.set_bandwidth(
                topology.devices[i], topology.devices[j], NETWORK_BANDWIDTH_Gbps
            )
    function, input_data = import_from_onnx(
        args.model_path,
        name="GPT-2",
        default_device=d0,
        function_output_names=set(["output1"]),
        parse_input_data=True,
    )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(
        "Here is some text to encode Hello World", add_special_tokens=True
    )
    input_ids = torch.tensor([[tokens] for _ in range(args.batch_size)])
    input_ids = to_numpy(input_ids)
    print(input_ids.shape)

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
    ex = SequentialExecutor("numpy")
    function = ex.infer_types(function, input_data)
    cpprint(function)

    function = gpt2_dhp_transform(
        function,
        args.dp_degree,
        args.hp_degree,
        args.pp_degree,
        topology.devices,
        args.num_microbatches,
    )
    function = ex.infer_types(function, input_data)
    cpprint(function)
    # output = ex.compute(function, input_data)
    """
    simulator = PostTypeInferenceSimulator(CostModel(topology))
    simulation = simulator.interpret(function, (v.type for v in function.inputs))

    op_costs = defaultdict(list)
    for event in simulation.trace:
        op_costs[event["name"]].append(event["dur"])
    for op_type in op_costs:
        print(f"{op_type}: {np.median(op_costs[op_type]) * 1e6} us")
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to ONNX model"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "-d", "--dp_degree", type=int, default=1, help="Data parallel degree"
    )
    parser.add_argument(
        "-t", "--hp_degree", type=int, default=1, help="Horizontal parallel degree"
    )
    parser.add_argument(
        "-p", "--pp_degree", type=int, default=1, help="Pipeline parallel degree"
    )
    parser.add_argument(
        "-k", "--num_microbatches", type=int, default=1, help="Num microbatches"
    )
    args = parser.parse_args()
    main(args)
