import argparse
from collections import defaultdict
import numpy as np
import re
from transformers import GPT2Tokenizer
import torch

import dist_ir.backend.torch as torch_backend
from dist_ir.executor import (
    CostModel,
    infer_types,
    PostTypeInferenceSimulator,
    Simulator,
    SequentialExecutor,
)
from dist_ir.importer import import_from_onnx
from dist_ir.ir import cpprint, Device, FunctionMaker, Op, Topology, Value
from dist_ir.ir.type import Float32, Tensor
from dist_ir.transforms import (
    gpt2_dhp_transform,
    sanitize_unhashable_attributes,
    restore_unhashable_attributes,
)

NETWORK_BANDWIDTH_Gbps = 200


def _to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x


def _filter_extra_outputs(function):
    function, attribute_map = sanitize_unhashable_attributes(function)

    # Map from op to set of function output values.
    sinks = defaultdict(set)

    # Set the sink for each output producer op to be the output.
    for output in function.outputs:
        producer = function.producers[output]
        sinks[producer] = set([output])

    # Incrementally propogate the set of sinks for each op by iterating through
    # all ops in reverse topological order.
    ops = list(function.ops)[::-1]
    while len(ops) > 0:
        op = ops.pop(0)
        for output in op.outputs:
            for consumer in function.consumers[output]:
                sinks[op] = sinks[op].union(sinks[consumer])

    # Filter out ops with no sinks other than output1.
    filtered_ops = set()
    for op in sinks:
        if function.outputs[-1] not in sinks[op]:
            filtered_ops.add(op)
    filtered_function = FunctionMaker(name=function.name)
    value_map = {}
    for inp in function.inputs:
        v = filtered_function.add_input_value(inp.name, inp.type)
        value_map[inp] = v
    for op in function.ops:
        if op in filtered_ops:
            continue
        inputs = tuple(value_map[inp] for inp in op.inputs)
        new_op = Op(
            name=op.name,
            op_type=op.op_type,
            inputs=inputs,
            attributes=op.attributes,
            subfunctions=op.subfunctions,
            output_names=tuple(output.name for output in op.outputs),
            output_types=tuple(output.type for output in op.outputs),
        )
        filtered_function.ops.append(new_op)
        for orig_output, new_output in zip(op.outputs, new_op.outputs):
            value_map[orig_output] = new_output

    filtered_function = restore_unhashable_attributes(filtered_function, attribute_map)
    return filtered_function.finalize()


def _set_model_size(function, num_transformer_blocks):
    function, attribute_map = sanitize_unhashable_attributes(function)

    # Prepare a list of the existing Transformer blocks in the function.
    blocks = []
    cur_block = []
    cur_block_id = 0
    orig_block_id_map = {}
    for op in function.ops:
        orig_block_id_map[op] = cur_block_id
        cur_block.append(op)
        if op.op_type == "Gemm" and any(
            "mlp.c_proj.weight" in inp.name for inp in op.inputs
        ):
            blocks.append(cur_block)
            cur_block_id += 1
            cur_block = []
    final_ops = cur_block
    for op in final_ops:
        orig_block_id_map[op] = cur_block_id

    # Verify that all blocks other than the first block are identical.
    transformer_block = tuple(op.op_type for op in blocks[1])
    for i in range(2, len(blocks)):
        assert tuple(op.op_type for op in blocks[i]) == transformer_block

    # Initialize a new function using the Transformer blocks from the original function.
    # We discard any original blocks beyond the requested number of new blocks.
    transformed_function = FunctionMaker(name=function.name)

    # A map from values in the original function to values in the transformed function.
    value_map = {}

    # A map from values in the transformed function to a tuple of
    # 1) the op which produced the value and
    # 2) the index of this op in the list of block ops.
    producer_map = {}

    # Add inputs from the original function to the transformed function.
    for inp in function.inputs:
        # Only add inputs if they are used by blocks that will appear
        # in the transformed function.
        max_consumer_block_id = max(
            [orig_block_id_map[consumer] for consumer in function.consumers[inp]]
        )
        if (
            max_consumer_block_id < num_transformer_blocks
            or max_consumer_block_id == len(blocks)
        ):
            value_map[inp] = transformed_function.add_input_value(inp.name, inp.type)

    # A map from ops in the transformed function to block id.
    block_id_map = {}
    transformed_blocks = []
    for i in range(min(num_transformer_blocks, len(blocks))):
        cur_block = []
        for k, op in enumerate(blocks[i]):
            inputs = tuple(value_map[inp] for inp in op.inputs)
            new_op = Op(
                name=op.name,
                op_type=op.op_type,
                inputs=inputs,
                attributes=op.attributes,
                subfunctions=op.subfunctions,
                output_names=tuple(output.name for output in op.outputs),
                output_types=tuple(output.type for output in op.outputs),
            )
            transformed_function.ops.append(new_op)
            for orig_output, new_output in zip(op.outputs, new_op.outputs):
                value_map[orig_output] = new_output
                producer_map[new_output] = (new_op, k)
            cur_block.append(new_op)
            block_id_map[new_op] = i
        transformed_blocks.append(cur_block)

    # Add any additional Transformer blocks if necessary.
    for j in range(len(blocks), num_transformer_blocks):
        cur_block = []
        for k, op in enumerate(transformed_blocks[-1]):
            # Collect the inputs for the new op.
            inputs = []
            for inp in op.inputs:
                if inp in transformed_function.inputs:
                    if "weight" in inp.name or "bias" in inp.name:
                        block_id = re.search("h\.(\d+)\.", inp.name).group(1)
                        new_name = inp.name.replace(block_id, str(j))
                        inputs.append(
                            transformed_function.add_input_value(new_name, inp.type)
                        )
                    else:
                        inputs.append(inp)
                else:
                    producer, producer_op_id = producer_map[inp]
                    output_index = producer.outputs.index(inp)
                    if block_id_map[producer] == j - 2:
                        # If the input value was produced in the previous block,
                        # the input for the next block will come from the
                        # corresponding op in the current block.
                        inputs.append(
                            transformed_blocks[-1][producer_op_id].outputs[output_index]
                        )
                    elif block_id_map[producer] == j - 1:
                        # If the input value was produced in the current block,
                        # the input for the next block will come from earlier in
                        # the next block.
                        inputs.append(cur_block[producer_op_id].outputs[output_index])
                    else:
                        # There can be no input from any other block because each
                        # block is self-contained with the exception of function
                        # inputs and outputs from the immediately preceding block.
                        raise ValueError(
                            f"Op {op} in block {j-1} has an input from "
                            f"block {block_id_map[producer]}"
                        )
            # TODO: Update op name
            # TODO: Update output names
            new_op = Op(
                name=op.name,
                op_type=op.op_type,
                inputs=tuple(inputs),
                attributes=op.attributes,
                subfunctions=op.subfunctions,
                output_names=tuple(output.name for output in op.outputs),
                output_types=tuple(output.type for output in op.outputs),
            )
            for output in new_op.outputs:
                producer_map[output] = (new_op, k)
            transformed_function.ops.append(new_op)
            cur_block.append(new_op)
            block_id_map[new_op] = j
        transformed_blocks.append(cur_block)

    # Add the final ops.
    for op, transformed_op in zip(blocks[-1], transformed_blocks[-1]):
        for output, transformed_output in zip(op.outputs, transformed_op.outputs):
            value_map[output] = transformed_output
    for op in final_ops:
        inputs = [value_map[inp] for inp in op.inputs]
        new_op = Op(
            name=op.name,
            op_type=op.op_type,
            inputs=tuple(inputs),
            attributes=op.attributes,
            subfunctions=op.subfunctions,
            output_names=tuple(output.name for output in op.outputs),
            output_types=tuple(output.type for output in op.outputs),
        )
        transformed_function.ops.append(new_op)
        for output, transformed_output in zip(op.outputs, new_op.outputs):
            value_map[output] = transformed_output

    transformed_function = restore_unhashable_attributes(
        transformed_function, attribute_map
    )

    return transformed_function.finalize()


def import_function_and_get_input_data(
    model_path, batch_size, num_transformer_blocks, default_device
):
    function, input_data_map = import_from_onnx(
        model_path,
        name="GPT-2",
        default_device=default_device,
        parse_input_data=True,
    )

    function = _filter_extra_outputs(function)
    function = _set_model_size(function, num_transformer_blocks)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(
        "Here is some text to encode Hello World", add_special_tokens=True
    )
    input_ids = torch.tensor([[tokens] for _ in range(batch_size)])
    input_ids = _to_numpy(input_ids)
    input_data = [input_ids] + list(input_data_map.values())
    # If any extra input weights were added, use the last occurence of the
    # corresponding weights in the original function as the initial weights.
    # This minimizes risk of numerical stability issues.
    if len(input_data) < len(function.inputs):
        extra_weight_map = {}
        for inp in input_data_map:
            base_input_name = re.sub("h\.(\d+)", "", inp.name)
            extra_weight_map[base_input_name] = input_data_map[inp]
        input_data += [
            extra_weight_map[re.sub("h\.(\d+)", "", inp.name)]
            for inp in function.inputs[len(input_data) :]
        ]
    return function, input_data


def transform(
    function,
    input_data,
    topology,
    dp_degree,
    hp_degree,
    pp_degree,
    num_microbatches,
    device_throughput=1.38e13,
    dram_bandwidth=7e11,
    network_bandwidth=77,
):
    world_size = dp_degree * hp_degree * pp_degree
    for i in range(1, world_size + 1):
        topology.add_device(
            "gpu", throughput=device_throughput, dram_bandwidth=dram_bandwidth
        )
        for j in range(0, i):
            if j == 0:
                topology.set_bandwidth(
                    topology.devices[i], topology.devices[j], network_bandwidth
                )
            else:
                topology.set_bandwidth(
                    topology.devices[i], topology.devices[j], network_bandwidth
                )
    init_function, transformed_function = gpt2_dhp_transform(
        function,
        dp_degree,
        hp_degree,
        pp_degree,
        topology.devices,
        num_microbatches,
    )
    # Manual adjustments for horizontal parallelism
    for i in range(len(input_data)):
        if input_data[i].shape == (1,) and (
            input_data[i][0] == 2304 or input_data[i][0] == 3072
        ):
            input_data[i] = np.array([input_data[i][0] // hp_degree])
    ex = SequentialExecutor("numpy")
    init_function = ex.infer_types(
        init_function,
        input_data,
        input_devices=[topology.devices[0] for _ in range(len(input_data))],
    )
    initialized_input_data = ex.compute(init_function, input_data)
    transformed_function = ex.infer_types(
        transformed_function,
        initialized_input_data,
        [output.type.device for output in init_function.outputs],
    )
    return init_function, transformed_function, initialized_input_data


def simulate(function, input_data, topology):
    input_types = (v.type for v in function.inputs)
    simulator = PostTypeInferenceSimulator(CostModel(topology))
    simulation = simulator.interpret(function, input_types)
    return simulation


def run_pytorch(function, input_data, world_size, use_gpu=True):
    pytorch_input_data = [torch.tensor(x) for x in input_data]
    if use_gpu and world_size > torch.cuda.device_count():
        raise ValueError(
            f"Specified world size is {world_size}, but only "
            f"{torch.cuda.device_count()} GPUs available"
        )
    per_rank_outputs, runtimes = torch_backend.run_pytorch(
        function,
        pytorch_input_data,
        use_gpu=use_gpu,
        run_type_inference=False,
    )
    return per_rank_outputs, runtimes


def main(args):
    topology = Topology()
    d0 = topology.add_device("gpu")
    function, input_data = import_function_and_get_input_data(
        args.model_path,
        batch_size=args.batch_size,
        num_transformer_blocks=args.num_transformer_blocks,
        default_device=d0,
    )
    ex = SequentialExecutor("numpy")
    function = ex.infer_types(
        function,
        input_data,
        input_devices=[topology.devices[0] for _ in range(len(input_data))],
    )
    init_function, transformed_function, initialized_input_data = transform(
        function,
        input_data,
        topology,
        args.dp_degree,
        args.hp_degree,
        args.pp_degree,
        args.num_microbatches,
    )
    if args.backend == "simulate":
        simulation = simulate(transformed_function, initialized_input_data, topology)
        if args.trace_file is not None:
            simulation.dump_chrome_trace(args.trace_file)
        distributed_running_time = max(
            [simulation.timestamps[d] for d in simulation.timestamps]
        )
        print(
            f"Throughput: {args.batch_size / distributed_running_time:.2f} "
            f"samples/second"
        )
    elif args.backend == "pytorch":
        world_size = args.dp_degree * args.hp_degree * args.pp_degree
        per_rank_outputs, runtimes = run_pytorch(
            transformed_function, initialized_input_data, world_size, args.use_gpu
        )
        print(
            f"Throughput: {args.batch_size / np.median(runtimes[-1]):.2f} "
            f"samples/second"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to GPT-2 ONNX model"
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
    parser.add_argument(
        "--num_transformer_blocks", type=int, default=12, help="Num transformer blocks"
    )
    parser.add_argument(
        "--backend",
        choices=["simulate", "pytorch"],
        default="simulate",
        help="Operation to run",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Use GPU with PyTorch backend",
    )
    parser.add_argument("--trace_file", type=str, default=None, help="Trace file")
    args = parser.parse_args()
    main(args)
