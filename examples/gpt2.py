from collections import defaultdict
import numpy as np
import re
from transformers import GPT2Tokenizer
import torch

import dist_ir.backend.torch as torch_backend
from dist_ir.executor import (
    CostModel,
    infer_types,
    sequentially_execute,
    Simulator,
    ConcreteValue,
)
from dist_ir.importer import import_from_onnx
from dist_ir.ir import FunctionMaker, Op, Value, get_uniform_topology
from dist_ir.ir.type import Float16, Float32, Tensor, Type, abstract_values
from dist_ir.transforms import (
    gpt2_dhp_transform,
    sanitize_unhashable_attributes,
    restore_unhashable_attributes,
)
from dist_ir.transforms.gpt2_dhp_transform import check_params, update_attributes

from .parser import Parser


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


def _set_model_size(function, n_layer, n_head, d_embd, dtype):
    dist_ir_dtype = Float32 if dtype == "fp32" else Float16
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
    inputs_to_remove = []
    for i, inp in enumerate(function.inputs):
        # Only add inputs if they are used by blocks that will appear
        # in the transformed function.
        max_consumer_block_id = max(
            [orig_block_id_map[consumer] for consumer in function.consumers[inp]]
        )
        if max_consumer_block_id < n_layer or max_consumer_block_id == len(blocks):
            # Resize the weights and biases according to the specified parameters.
            shape = inp.type.shape
            if inp.name == "wte.weight":
                vocab_size = inp.type.shape[0]
                shape = (vocab_size, d_embd)
                typ = Tensor(shape=shape, device=inp.type.device, dtype=dist_ir_dtype())
            elif inp.name == "wpe.weight":
                max_position_embeddings = inp.type.shape[0]
                shape = (max_position_embeddings, d_embd)
                typ = Tensor(shape=shape, device=inp.type.device, dtype=dist_ir_dtype())
            elif (
                "ln_1.weight" in inp.name
                or "ln_1.bias" in inp.name
                or "ln_2.weight" in inp.name
                or "ln_2.bias" in inp.name
                or "ln_f.weight" in inp.name
                or "ln_f.bias" in inp.name
            ):
                shape = (d_embd,)
            elif "c_attn.weight" in inp.name:
                shape = (d_embd, 3 * d_embd)
            elif "c_attn.bias" in inp.name:
                shape = (3 * d_embd,)
            elif "attn.c_proj.weight" in inp.name:
                shape = (d_embd, d_embd)
            elif "attn.c_proj.bias" in inp.name:
                shape = (d_embd,)
            elif "c_fc.weight" in inp.name:
                shape = (d_embd, 4 * d_embd)
            elif "c_fc.bias" in inp.name:
                shape = (4 * d_embd,)
            elif "mlp.c_proj.weight" in inp.name:
                shape = (4 * d_embd, d_embd)
            elif "mlp.c_proj.bias" in inp.name:
                shape = (d_embd,)
            if shape != inp.type.shape:
                typ = Tensor(shape=shape, device=inp.type.device, dtype=dist_ir_dtype())
            else:
                typ = inp.type
            value_map[inp] = transformed_function.add_input_value(inp.name, typ)
        else:
            inputs_to_remove.append(i)

    # A map from ops in the transformed function to block id.
    block_id_map = {}

    # Counters to keep track of the maximum op and output IDs seen so far.
    max_op_id = -1
    max_output_id = -1

    # Add ops from the original Transformer blocks to the new function.
    transformed_blocks = []
    for i in range(min(n_layer, len(blocks))):
        cur_block = []
        for k, op in enumerate(blocks[i]):
            max_op_id = max(max_op_id, int(re.match(r".*_(\d+)", op.name).group(1)))
            inputs = tuple(value_map[inp] for inp in op.inputs)
            if op.op_type == "Split" or op.op_type == "Constant":
                attributes = update_attributes(
                    op.op_type,
                    op.attributes,
                    attribute_map,
                    old_d_embd=768,
                    new_d_embd=d_embd,
                    old_n_head=12,
                    new_n_head=n_head,
                )
            else:
                attributes = op.attributes
            new_op = Op(
                name=op.name,
                op_type=op.op_type,
                inputs=inputs,
                attributes=attributes,
                subfunctions=op.subfunctions,
                output_names=tuple(output.name for output in op.outputs),
                output_types=tuple(output.type for output in op.outputs),
            )
            transformed_function.ops.append(new_op)
            for orig_output, new_output in zip(op.outputs, new_op.outputs):
                if (
                    "query" not in orig_output.name
                    and "key" not in orig_output.name
                    and "value" not in orig_output.name
                ):
                    max_output_id = max(
                        max_output_id,
                        int(re.match(r"(\d+)", orig_output.name).group(1)),
                    )
                value_map[orig_output] = new_output
                producer_map[new_output] = (new_op, k)
            cur_block.append(new_op)
            block_id_map[new_op] = i
        transformed_blocks.append(cur_block)

    # Add any additional Transformer blocks if necessary.
    for j in range(len(blocks), n_layer):
        cur_block = []
        for k, op in enumerate(transformed_blocks[-1]):
            # Collect the inputs for the new op.
            inputs = []
            for inp in op.inputs:
                if inp in transformed_function.inputs:
                    if "weight" in inp.name or "bias" in inp.name:
                        block_id = re.search(r"h\.(\d+)\.", inp.name).group(1)
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
            if op.op_type == "Split":
                assert "query" in op.outputs[0].name
                assert "key" in op.outputs[1].name
                assert "value" in op.outputs[2].name
                output_names = (f"query.{j}", f"key.{j}", f"value.{j}")
            else:
                output_names = []
                for _ in range(len(op.outputs)):
                    output_names.append(str(max_output_id))
                    max_output_id += 1
                output_names = tuple(output_names)
            new_op = Op(
                name=f"{op.op_type}_{max_op_id}",
                op_type=op.op_type,
                inputs=tuple(inputs),
                attributes=op.attributes,
                subfunctions=op.subfunctions,
                output_names=output_names,
                output_types=tuple(output.type for output in op.outputs),
            )
            max_op_id += 1
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

    return transformed_function.finalize(), inputs_to_remove


def _get_stats(function, input_types):
    parameter_count = 0
    model_size = 0
    for inp, typ in zip(function.inputs, input_types):
        if "weight" in inp.name or "bias" in inp.name:
            parameter_count += np.prod(typ.shape)
            model_size += typ.size()

    if parameter_count >= 1e3 and parameter_count < 1e6:
        parameter_count_str = f"{parameter_count / 1e3:.2f}K"
    elif parameter_count >= 1e6 and parameter_count < 1e9:
        parameter_count_str = f"{parameter_count / 1e6:.2f}M"
    elif parameter_count >= 1e9:
        parameter_count_str = f"{parameter_count / 1e9:.2f}B"
    else:
        parameter_count_str = str(parameter_count)

    if model_size >= 1e3 and model_size < 1e6:
        model_size_str = f"{model_size / 1e3:.2f} KB"
    elif model_size >= 1e6 and model_size < 1e9:
        model_size_str = f"{model_size / 1e6:.2f} MB"
    elif model_size >= 1e9:
        model_size_str = f"{model_size / 1e9:.2f} GB"
    else:
        model_size_str = str(model_size)

    return parameter_count, model_size, parameter_count_str, model_size_str


def import_function_and_get_input_data(
    model_path,
    default_device,
    dtype,
    use_real_weights=False,
):
    is_input_or_weight = lambda x: "input" in x or "weight" in x or "bias" in x

    function, input_data_map = import_from_onnx(
        model_path,
        name="GPT-2",
        default_device=default_device,
        parse_input_data=True,
    )

    function = _filter_extra_outputs(function)

    for inp in input_data_map:
        if is_input_or_weight(inp.name):
            if not use_real_weights:
                if dtype == "fp16" and isinstance(inp.type.dtype, Float32):
                    input_data_map[inp] = Tensor(
                        shape=inp.type.shape,
                        dtype=Float16(inp.type.dtype.device),
                        device=inp.type.device,
                    )
                else:
                    input_data_map[inp] = inp.type
            elif dtype == "fp16" and input_data_map[inp].dtype == np.float32:
                input_data_map[inp] = input_data_map[inp].astype(np.float16)
    input_data = list(input_data_map.values())

    return function, input_data


def resize_function_and_input_data(
    function, input_data, n_layer, n_head, d_embd, dtype
):
    function, inputs_to_remove = _set_model_size(
        function, n_layer, n_head, d_embd, dtype
    )

    # If we shrunk the model, remove any unnecessary inputs.
    for i in inputs_to_remove[::-1]:
        input_data.pop(i - 1)

    # Update the input data if any weight shapes were changed.
    for i in range(len(input_data)):
        inp = function.inputs[i + 1]
        old_shape = input_data[i].shape
        if old_shape != inp.type.shape:
            assert "weight" in inp.name or "bias" in inp.name
            if isinstance(input_data[i], np.ndarray):
                # Zero-pad the new weights.
                new_tensor = np.zeros(inp.type.shape, dtype=input_data[i].dtype)
                if len(old_shape) == 1:
                    new_tensor[: old_shape[0]] = input_data[i]
                elif len(old_shape) == 2:
                    new_tensor[: old_shape[0], : old_shape[1]] = input_data[i]
                input_data[i] = new_tensor
            else:
                assert isinstance(input_data[i], Tensor)
                input_data[i] = inp.type
        elif old_shape == (1,):
            assert "weight" not in inp.name and "bias" not in inp.name
            if input_data[i][0] == 768:
                input_data[i] = np.array([d_embd], dtype=input_data[i].dtype)
            elif input_data[i][0] == 768 * 3:
                input_data[i] = np.array([d_embd * 3], dtype=input_data[i].dtype)
            elif input_data[i][0] == 768 * 4:
                input_data[i] = np.array([d_embd * 4], dtype=input_data[i].dtype)
            elif input_data[i][0] == 12:
                input_data[i] = np.array([n_head], dtype=input_data[i].dtype)

    # If any extra input weights were added, use the last occurence of the
    # corresponding weights in the original function as the initial weights.
    # This minimizes risk of numerical stability issues.
    if len(input_data) < len(function.inputs) - 1:
        extra_weight_map = {}
        for i, inp in enumerate(function.inputs[1 : 1 + len(input_data)]):
            base_input_name = re.sub(r"h\.(\d+)", "", inp.name)
            extra_weight_map[base_input_name] = input_data[i]
        input_data += [
            extra_weight_map[re.sub(r"h\.(\d+)", "", inp.name)]
            for inp in function.inputs[1 + len(input_data) :]
        ]
    return function, input_data


def create_input_ids(batch_size):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(
        "Here is some text to encode Hello World", add_special_tokens=True
    )
    input_ids = torch.tensor([[tokens] for _ in range(batch_size)])
    return _to_numpy(input_ids)


def _update_input_data_for_hp(
    input_data, function, d_embd, n_head, hp_degree, use_real_weights
):
    for i, inp in enumerate(function.inputs):
        if input_data[i].shape == (1,):
            if input_data[i][0] == d_embd * 3:
                input_data[i] = np.array(
                    [d_embd * 3 // hp_degree], dtype=input_data[i].dtype
                )
            elif input_data[i][0] == d_embd * 4:
                input_data[i] = np.array(
                    [d_embd * 4 // hp_degree], dtype=input_data[i].dtype
                )
            elif input_data[i][0] == n_head:
                input_data[i] = np.array(
                    [n_head // hp_degree], dtype=input_data[i].dtype
                )
        elif use_real_weights and "c_proj.bias" in inp.name:
            input_data[i] = np.copy(input_data[i]) / hp_degree


def transform(
    function,
    input_data,
    topology,
    dp_degree,
    hp_degree,
    pp_degree,
    num_microbatches,
    d_embd,
    n_head,
    skip_allgathers=False,
    use_real_weights=False,
):
    if hp_degree > 1:
        _update_input_data_for_hp(
            input_data, function, d_embd, n_head, hp_degree, use_real_weights
        )
    world_size = dp_degree * hp_degree * pp_degree
    init_function, transformed_function = gpt2_dhp_transform(
        function,
        dp_degree,
        hp_degree,
        pp_degree,
        topology.devices[: world_size + 1],
        num_microbatches,
        d_embd,
        n_head,
        skip_allgathers=skip_allgathers,
    )
    wrapped_input_data = []
    for v in input_data:
        if isinstance(v, Type):
            wrapped_input_data.append(v)
        else:
            wrapped_input_data.append(ConcreteValue(v, topology.devices[0]))
    initialized_input_data = sequentially_execute(init_function, wrapped_input_data)
    return init_function, transformed_function, initialized_input_data


def get_transformed_function_and_input_data(
    model_path,
    dtype,
    device_throughput,
    dram_bandwidth,
    kernel_launch_overhead,
    network_bandwidth,
    batch_size,
    dp_degree,
    hp_degree,
    pp_degree,
    num_microbatches,
    n_layer,
    n_head,
    d_embd,
    skip_allgathers=False,
    use_real_weights=False,
    print_stats=False,
):
    world_size = dp_degree * hp_degree * pp_degree
    topology = get_uniform_topology(
        world_size,
        device_throughput,
        dram_bandwidth,
        kernel_launch_overhead,
        network_bandwidth,
    )

    function, input_data = import_function_and_get_input_data(
        model_path,
        default_device=topology.devices[0],
        dtype=dtype,
        use_real_weights=use_real_weights,
    )

    function, input_data = resize_function_and_input_data(
        function, input_data, n_layer, n_head, d_embd, dtype
    )

    input_ids = create_input_ids(batch_size)
    input_data = [input_ids] + input_data

    if print_stats:
        parameter_count, model_size, parameter_count_str, model_size_str = _get_stats(
            function, input_data
        )
        print("Parameter count:", parameter_count_str)
        print("Model size:", model_size_str)

    init_function, transformed_function, initialized_input_data = transform(
        function,
        input_data,
        topology,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
        d_embd,
        n_head,
        skip_allgathers=skip_allgathers,
        use_real_weights=use_real_weights,
    )

    return transformed_function, initialized_input_data, topology


def simulate(function, input_data, topology, allreduce_parameters=None):
    simulator = Simulator(CostModel(topology, allreduce_parameters))
    simulation = simulator.simulate(function, input_data)
    return simulation


def run_pytorch(
    function, input_data, world_size, use_gpu=False, debug_stacktrace=False
):
    # TODO: Move this to a utils file
    def _resolve_dtype(dtype):
        if dtype == np.int32:
            return torch.int32
        elif dtype == np.int64:
            return torch.int64
        elif dtype == np.float16:
            return torch.float16
        elif dtype == np.float32:
            return torch.float32
        else:
            raise NotImplementedError(dtype)

    is_input_or_weight = lambda x: "input" in x or "weight" in x or "bias" in x

    input_types = abstract_values(
        input_data,
        tuple(
            Tensor if is_input_or_weight(function.inputs[i].name) else ConcreteValue
            for i in range(len(input_data))
        ),
    )
    pytorch_input_data = [
        torch.tensor(x.val, dtype=_resolve_dtype(x.val.dtype)) for x in input_data
    ]

    if use_gpu and world_size > torch.cuda.device_count():
        raise ValueError(
            f"Specified world size is {world_size}, but only "
            f"{torch.cuda.device_count()} GPUs available"
        )
    per_rank_outputs, runtimes = torch_backend.run_pytorch(
        function,
        pytorch_input_data,
        input_types=input_types,
        use_gpu=use_gpu,
        num_warmup=5,
        num_repetitions=10,
        debug_stacktrace=debug_stacktrace,
    )
    return per_rank_outputs, runtimes


def main(args):
    check_params(
        args.batch_size,
        args.dp_degree,
        args.hp_degree,
        args.pp_degree,
        args.num_microbatches,
        args.n_head,
        args.d_embd,
    )

    if args.backend == "pytorch":
        args.use_real_weights = True

    (
        transformed_function,
        initialized_input_data,
        topology,
    ) = get_transformed_function_and_input_data(
        args.model_path,
        args.dtype,
        args.device_throughput,
        args.dram_bandwidth,
        args.kernel_launch_overhead,
        args.network_bandwidth,
        args.batch_size,
        args.dp_degree,
        args.hp_degree,
        args.pp_degree,
        args.num_microbatches,
        args.n_layer,
        args.n_head,
        args.d_embd,
        args.skip_allgathers,
        args.use_real_weights,
        print_stats=True,
    )

    if args.backend == "simulate":
        simulation = simulate(transformed_function, initialized_input_data, topology)
        if args.trace_file is not None:
            simulation.dump_chrome_trace(args.trace_file)
        simulation.print_summary(args.batch_size)
    elif args.backend == "pytorch":
        world_size = args.dp_degree * args.hp_degree * args.pp_degree
        per_rank_outputs, runtimes = run_pytorch(
            transformed_function,
            initialized_input_data,
            world_size,
            use_gpu=args.use_gpu,
            debug_stacktrace=args.debug_stacktrace,
        )
        print(f"Latency: {np.median(runtimes[-1])*1000:.2f} ms")
        print(
            f"Throughput: {args.batch_size / np.median(runtimes[-1]):.2f} "
            f"samples/second"
        )


if __name__ == "__main__":
    parser = Parser("GPT2 Inference")
    parser.add_parallelism_config_arguments()
    parser.add_simulation_config_arguments()
    parser.add_backend_config_arguments()
    parser.add_execution_mode_config_arguments()
    parser.add_gpt2_model_path_config_arguments()
    parser.add_simulation_output_config_arguments()
    parser.add_argument("--n_layer", type=int, default=12, help="Num hidden layers")
    parser.add_argument(
        "--n_head",
        type=int,
        default=12,
        help="Number of attention heads for each attention layer",
    )
    parser.add_argument("--d_embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument(
        "--use_real_weights",
        action="store_true",
        default=False,
        help="Use real weights",
    )
    args = parser.parse_args()
    main(args)
