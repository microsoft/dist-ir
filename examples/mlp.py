import argparse
from collections import defaultdict
import numpy as np
import re
import torch

from dist_ir.ir import FunctionMaker, Topology, get_uniform_topology, Value
from dist_ir.ir.type import Int32, Float16, Float32, Tensor, abstract_values
from dist_ir.executor import (
    CostModel,
    Simulator,
    ConcreteValue,
    infer_types,
    sequentially_execute,
)
from dist_ir.transforms import mlp_dhp_transform
from .parser import Parser
import dist_ir.backend.torch as torch_backend


def get_typed_input_values(inputs, batch_size, input_dim, output_dim):
    # TODO: Add types for weights as well?
    typed_inputs = list(inputs)
    # Update x and z to use the selected batch size
    typed_inputs[0] = Value(
        typed_inputs[0].name,
        Tensor(
            shape=(batch_size, input_dim),
            dtype=typed_inputs[0].type.dtype,
            device=typed_inputs[0].type.device,
        ),
    )
    typed_inputs[1] = Value(
        typed_inputs[1].name,
        Tensor(
            shape=(batch_size, output_dim),
            dtype=typed_inputs[1].type.dtype,
            device=typed_inputs[1].type.device,
        ),
    )
    # Add value for batch size
    typed_inputs[2] = Value(
        typed_inputs[2].name, Int32(device=typed_inputs[2].type.device)
    )
    return tuple(typed_inputs)


def get_input_data(inputs, batch_size, input_dim, output_dim, device, dtype):
    input_data = []
    x = np.random.normal(0, 0.02, size=(batch_size, input_dim))
    z = np.random.normal(0, 0.02, size=(batch_size, output_dim))
    n = np.int64(batch_size)
    weights = [np.random.normal(0, 0.02, size=inp.type.shape) for inp in inputs[3:]]
    input_data = [x, z, n] + weights
    input_data = [v.astype(dtype) if i != 2 else v for i, v in enumerate(input_data)]
    input_data = [ConcreteValue(v, device) for v in input_data]
    assert len(input_data) == len(inputs)
    return input_data


def mlp(input_dim, hidden_dim, output_dim, num_hidden_layers, device, dtype):
    function = FunctionMaker(name="mlp")
    x = function.add_input_value(
        "x",
        Tensor(dtype=dtype(), shape=None, device=device),
    )
    z = function.add_input_value(
        "z",
        Tensor(dtype=dtype(), shape=None, device=device),
    )
    n = function.add_input_value("n", Int32(device=device))
    weights = []
    w = function.add_input_value(
        "wA",
        Tensor(dtype=dtype(), shape=(input_dim, hidden_dim), device=device),
    )
    weights.append(w)
    for i in range(1, num_hidden_layers - 1):
        w = function.add_input_value(
            f"w{chr(ord('A')+i)}",
            Tensor(dtype=dtype(), shape=(hidden_dim, hidden_dim), device=device),
        )
        weights.append(w)
    w = function.add_input_value(
        f"w{chr(ord('A')+i+1)}",
        Tensor(dtype=dtype(), shape=(hidden_dim, output_dim), device=device),
    )
    weights.append(w)

    a = x
    for i, weight in enumerate(weights):
        y = function.add_op("MatMul", inputs=[a, weight], output_names=[f"y{i}"])
        a = function.add_op("Relu", inputs=[y], output_names=[f"a{i}"])

    l = function.add_op("Loss", inputs=[a, z, n], output_names=["l"])
    dl = function.add_op(
        "LossGrad",
        inputs=[a, z, n],
        output_names=["dl"],
    )

    dy = dl
    for i, weight in enumerate(weights[::-1]):
        i = len(weights) - i - 1
        da = function.add_op(
            "ReluGrad",
            inputs=[function.ops[2 * i + 1].inputs[0], dy],
            output_names=[f"da{i}"],
        )
        dy, dw = function.add_op(
            "MatMulGrad",
            inputs=[function.ops[2 * i].inputs[0], weights[i], da],
            output_names=[f"dy{i}", f"dw{chr(ord('A')+i)}"],
        )
    return function.finalize()


def mlp_inference(
    batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, device, dtype
):
    function = FunctionMaker(name="mlp")
    weights = []
    for i in range(num_hidden_layers - 1):
        w = function.add_input_value(
            f"w{chr(ord('A')+i)}",
            Tensor(dtype=dtype(), shape=(input_dim, hidden_dim), device=device),
        )
        weights.append(w)
    w = function.add_input_value(
        f"w{chr(ord('A')+i+1)}",
        Tensor(dtype=dtype(), shape=(hidden_dim, output_dim), device=device),
    )
    weights.append(w)
    x = function.add_input_value(
        "x",
        Tensor(dtype=dtype(), shape=(batch_size, input_dim), device=device),
    )

    a = x
    for i, weight in enumerate(weights):
        y = function.add_op("MatMul", inputs=[a, weight], output_names=[f"y{i}"])
        a = function.add_op("Relu", inputs=[y], output_names=[f"a{i}"])

    return function.finalize()


def mlp_inference_dp(
    batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, devices, dtype
):
    num_devices = len(devices)
    assert batch_size % num_devices == 0
    function = FunctionMaker(name="mlp")
    weights = {}
    x = {}
    for d in devices:
        for i in range(num_hidden_layers - 1):
            weights[i, d] = function.add_input_value(
                f"w{chr(ord('A')+i)}_{d.device_id}",
                Tensor(dtype=dtype(), shape=(input_dim, hidden_dim), device=d),
            )
        weights[num_hidden_layers - 1, d] = function.add_input_value(
            f"w{chr(ord('A')+i+1)}_{d.device_id}",
            Tensor(dtype=dtype(), shape=(hidden_dim, output_dim), device=d),
        )
        x[d] = function.add_input_value(
            f"x_{d.device_id}",
            Tensor(
                dtype=dtype(), shape=(batch_size // num_devices, input_dim), device=d
            ),
        )

    a = x
    for i in range(num_hidden_layers):
        for d in devices:
            y = function.add_op(
                "MatMul",
                inputs=[a[d], weights[i, d]],
                output_names=[f"y{i}_{d.device_id}"],
            )
            a[d] = function.add_op(
                "Relu", inputs=[y], output_names=[f"a{i}_{d.device_id}"]
            )

    return function.finalize()


def add_optimizer_ops(function):
    function = function.to_function_maker()
    hp_group_pattern = r"hp\_(.+?(?=\_))"

    all_hp_groups = []
    for output in function.outputs:
        if "dw" in output.name:
            match = re.search(hp_group_pattern, output.name)
            if match is not None and match.group(1) != "all":
                hp_group = tuple([int(x) for x in match.group(1).split(",")])
                all_hp_groups.append(hp_group)
    if len(all_hp_groups) > 1:
        all_hp_groups = sorted(set(all_hp_groups), key=lambda x: x[0])

    weight_map = defaultdict(lambda: {})
    for inp in function.inputs:
        if inp.name[0] != "w":
            continue
        w = inp
        name = w.name.split("_")[0]
        match = re.search(r"dp_(\d+)", w.name)
        dp = int(match.group(1)) if match is not None else 0
        match = re.search(r"hp_(\d+)", w.name)
        hp = int(match.group(1)) if match is not None else 0
        pp = w.type.device.device_id
        weight_map[(dp, hp, pp)][name] = w

    gradient_map = defaultdict(lambda: {})
    for output in function.outputs:
        if "dw" not in output.name:
            continue
        dw = output
        name = dw.name.split("_")[0][1:]
        dp = 0 if "dp_all" not in dw.name else int(dw.name.split("_")[-1])
        match = re.search(hp_group_pattern, dw.name)
        if match is not None and match.group(1) != "all":
            hp_group = tuple([int(x) for x in match.group(1).split(",")])
            hp = all_hp_groups.index(hp_group)
        else:
            hp = 0
        pp = dw.type.device.device_id
        gradient_map[(dp, hp, pp)][name] = dw

    if sorted(weight_map.keys()) != sorted(gradient_map.keys()):
        raise ValueError(f"Devices do not match for weights and gradients")

    for device in weight_map:
        weight_keys = sorted(weight_map[device].keys())
        gradient_keys = sorted(gradient_map[device].keys())
        assert weight_keys == gradient_keys
        weights = [weight_map[device][k] for k in weight_keys]
        gradients = [gradient_map[device][k] for k in gradient_keys]

        function.add_op(
            op_type="SGDOptimizer",
            inputs=(weights + gradients),
            attributes={"lr": 0},
            output_names=[f"{w.name}'" for w in weights],
        )

    return function.finalize()


# TODO: De-duplicate this function with examples/gpt2.py
def get_stats(function):
    parameter_count = 0
    model_size = 0
    for inp in function.inputs:
        if "w" in inp.name:
            parameter_count += np.prod(inp.type.shape)
            model_size += inp.type.size()

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


def simulate(function, input_types, topology, allreduce_parameters=None):
    simulator = Simulator(CostModel(topology, allreduce_parameters))
    simulation = simulator.simulate(function, input_types)
    return simulation


def run_pytorch(function, input_data, world_size, use_gpu=torch.cuda.is_available()):
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

    if use_gpu and world_size > torch.cuda.device_count():
        raise ValueError(
            f"Specified world size is {world_size}, but only "
            f"{torch.cuda.device_count()} GPUs available"
        )
    pytorch_input_data = [
        torch.tensor(x.val, dtype=_resolve_dtype(x.val.dtype))
        if isinstance(x.val, np.ndarray)
        else torch.tensor(x.val, dtype=torch.int32)
        for x in input_data
    ]
    input_types = abstract_values(
        input_data,
        tuple(
            Tensor if isinstance(input_data[i].val, np.ndarray) else Int32
            for i in range(len(input_data))
        ),
    )
    per_rank_outputs, runtimes = torch_backend.run_pytorch(
        function,
        pytorch_input_data,
        input_types=input_types,
        use_gpu=use_gpu,
        num_warmup=5,
        num_repetitions=10,
    )
    return per_rank_outputs, runtimes


def run_mlp(
    phase,
    backend,
    dtype,
    use_gpu,
    batch_size,
    input_dim,
    hidden_dim,
    output_dim,
    num_hidden_layers,
    dp_degree,
    hp_degree,
    pp_degree,
    num_microbatches,
    device_throughput,
    dram_bandwidth,
    kernel_launch_overhead,
    network_bandwidth,
    trace_file,
    skip_allgathers=False,
    verbose=False,
):
    dist_ir_dtype = Float32 if dtype == "fp32" else Float16
    numpy_dtype = np.float32 if dtype == "fp32" else np.float16
    world_size = dp_degree * hp_degree * pp_degree
    topology = get_uniform_topology(
        world_size,
        device_throughput,
        dram_bandwidth,
        kernel_launch_overhead,
        network_bandwidth,
    )

    if phase == "training":
        fn = mlp(
            input_dim,
            hidden_dim,
            output_dim,
            num_hidden_layers,
            topology.devices[0],
            dist_ir_dtype,
        )
    elif phase == "inference":
        fn = mlp_inference(
            input_dim,
            hidden_dim,
            output_dim,
            num_hidden_layers,
            topology.devices[0],
            dist_ir_dtype,
        )

    if verbose:
        parameter_count, model_size, parameter_count_str, model_size_str = get_stats(fn)
        print("Parameter count:", parameter_count_str)
        print("Model size:", model_size_str)

    if backend == "pytorch":
        input_data = get_input_data(
            fn.inputs,
            batch_size,
            input_dim,
            output_dim,
            topology.devices[0],
            numpy_dtype,
        )

    if world_size > 1:
        init_fn, transformed_fn = mlp_dhp_transform(
            fn,
            dp_degree,
            hp_degree,
            pp_degree,
            num_microbatches,
            topology.devices,
            skip_allgathers=skip_allgathers,
        )
        typed_inputs = get_typed_input_values(
            init_fn.inputs, batch_size, input_dim, output_dim
        )
        init_fn = infer_types(init_fn, typed_inputs)
        transformed_fn = infer_types(transformed_fn, init_fn.outputs)
        input_types = tuple(output.type for output in init_fn.outputs)
        if backend == "pytorch":
            transformed_input_data = sequentially_execute(init_fn, input_data)
    else:
        typed_inputs = get_typed_input_values(
            fn.inputs, batch_size, input_dim, output_dim
        )
        fn = infer_types(fn, typed_inputs)
        transformed_fn = fn
        input_types = tuple(inp.type for inp in fn.inputs)
        if backend == "pytorch":
            transformed_input_data = input_data
    transformed_fn = add_optimizer_ops(transformed_fn)
    if backend == "simulate":
        simulation = simulate(transformed_fn, input_types, topology)
        if verbose:
            simulation.print_summary(batch_size=batch_size)
        if trace_file is not None:
            simulation.dump_chrome_trace(trace_file)
        return simulation
    elif backend == "pytorch":
        per_rank_outputs, runtimes = run_pytorch(
            transformed_fn, transformed_input_data, world_size, use_gpu
        )
        if verbose:
            latency = np.median(runtimes[-1])
            print(f"Latency: {latency}")
            print(f"Throughput: {batch_size / latency}")
        return per_rank_outputs, runtimes


def main(args):
    run_mlp(
        args.phase,
        args.backend,
        args.dtype,
        args.use_gpu,
        args.batch_size,
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        args.num_hidden_layers,
        args.dp_degree,
        args.hp_degree,
        args.pp_degree,
        args.num_microbatches,
        args.device_throughput,
        args.dram_bandwidth,
        args.kernel_launch_overhead,
        args.network_bandwidth,
        args.trace_file,
        args.skip_allgathers,
        args.verbose,
    )


if __name__ == "__main__":
    parser = Parser(description="MLP training and inference")
    parser.add_parallelism_config_arguments()
    parser.add_simulation_config_arguments()
    parser.add_execution_mode_config_arguments()
    parser.add_backend_config_arguments()
    parser.add_simulation_output_config_arguments()
    parser.add_global_output_config_arguments()
    parser.add_argument(
        "--phase", choices=["inference", "training"], default="training"
    )
    parser.add_argument("--input_dim", type=int, default=256, help="Input dim")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dim")
    parser.add_argument("--output_dim", type=int, default=256, help="Output dim")
    parser.add_argument(
        "--num_hidden_layers", type=int, default=16, help="# hidden layers"
    )
    args = parser.parse_args()
    main(args)
