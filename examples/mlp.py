import argparse
from collections import defaultdict
import numpy as np
import re

from dist_ir.ir import FunctionMaker, Topology
from dist_ir.ir.type import Float32, Tensor
from dist_ir.executor import CostModel, Simulator, infer_types
from dist_ir.transforms import mlp_dhp_transform


def mlp(
    batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, device, name="mlp"
):
    function = FunctionMaker(name=name)
    x = function.add_input_value(
        "x",
        Tensor(dtype=Float32(), shape=(batch_size, input_dim), device=device),
    )
    z = function.add_input_value(
        "z",
        Tensor(dtype=Float32(), shape=(batch_size, output_dim), device=device),
    )
    weights = []
    for i in range(num_hidden_layers - 1):
        if i == 0:
            w = function.add_input_value(
                f"w{chr(ord('A')+i)}",
                Tensor(dtype=Float32(), shape=(input_dim, hidden_dim), device=device),
            )
        else:
            w = function.add_input_value(
                f"w{chr(ord('A')+i)}",
                Tensor(dtype=Float32(), shape=(hidden_dim, hidden_dim), device=device),
            )
        weights.append(w)
    w = function.add_input_value(
        f"w{chr(ord('A')+i+1)}",
        Tensor(dtype=Float32(), shape=(hidden_dim, output_dim), device=device),
    )
    weights.append(w)

    a = x
    for i, weight in enumerate(weights):
        y = function.add_op("MatMul", inputs=[a, weight], output_names=[f"y{i}"])
        a = function.add_op("Relu", inputs=[y], output_names=[f"a{i}"])

    l = function.add_op(
        "Loss", inputs=[a, z], attributes={"N": batch_size}, output_names=["l"]
    )
    dl = function.add_op(
        "LossGrad",
        inputs=[a, z],
        attributes={"N": batch_size},
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
    batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, device
):
    function = FunctionMaker(name="mlp")
    weights = []
    for i in range(num_hidden_layers - 1):
        w = function.add_input_value(
            f"w{chr(ord('A')+i)}",
            Tensor(dtype=Float32(), shape=(input_dim, hidden_dim), device=device),
        )
        weights.append(w)
    w = function.add_input_value(
        f"w{chr(ord('A')+i+1)}",
        Tensor(dtype=Float32(), shape=(hidden_dim, output_dim), device=device),
    )
    weights.append(w)
    x = function.add_input_value(
        "x",
        Tensor(dtype=Float32(), shape=(batch_size, input_dim), device=device),
    )

    a = x
    for i, weight in enumerate(weights):
        y = function.add_op("MatMul", inputs=[a, weight], output_names=[f"y{i}"])
        a = function.add_op("Relu", inputs=[y], output_names=[f"a{i}"])

    return function.finalize()


def mlp_inference_dp(
    batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, devices
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
                Tensor(dtype=Float32(), shape=(input_dim, hidden_dim), device=d),
            )
        weights[num_hidden_layers - 1, d] = function.add_input_value(
            f"w{chr(ord('A')+i+1)}_{d.device_id}",
            Tensor(dtype=Float32(), shape=(hidden_dim, output_dim), device=d),
        )
        x[d] = function.add_input_value(
            f"x_{d.device_id}",
            Tensor(
                dtype=Float32(), shape=(batch_size // num_devices, input_dim), device=d
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
    hp_group_pattern = "hp\_(.+?(?=\_))"

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
        match = re.search("dp_(\d+)", w.name)
        dp = int(match.group(1)) if match is not None else 0
        match = re.search("hp_(\d+)", w.name)
        hp = int(match.group(1)) if match is not None else 0
        weight_map[(dp, hp)][name] = w

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
        gradient_map[(dp, hp)][name] = dw

    if sorted(weight_map.keys()) != sorted(gradient_map.keys()):
        import pdb

        pdb.set_trace()
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
            attributes={"lr": 1e-3},
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


# TODO: De-duplicate this function with examples/gpt2.py
def get_topology(
    world_size, device_throughput=1.4e13, dram_bandwidth=9e11, network_bandwidth=64
):
    topology = Topology()
    d0 = topology.add_device("gpu")
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
    return topology


def simulate(function, input_types, topology):
    simulator = Simulator(CostModel(topology))
    simulation = simulator.interpret(function, input_types)
    return simulation


def main(args):
    world_size = args.dp_degree * args.hp_degree * args.pp_degree
    topology = get_topology(
        world_size, args.device_throughput, args.dram_bandwidth, args.network_bandwidth
    )

    if args.mode == "training":
        function = mlp(
            args.batch_size,
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_hidden_layers,
            topology.devices[0],
        )
    elif args.mode == "inference":
        function = mlp_inference(
            args.batch_size,
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_hidden_layers,
            topology.devices[0],
        )

    parameter_count, model_size, parameter_count_str, model_size_str = get_stats(
        function
    )
    print("Parameter count:", parameter_count_str)
    print("Model size:", model_size_str)

    if world_size > 1:
        init_function, transformed_function = mlp_dhp_transform(
            function,
            args.dp_degree,
            args.hp_degree,
            args.pp_degree,
            args.num_microbatches,
            topology.devices,
        )
        init_function = infer_types(init_function, init_function.inputs)
        input_types = tuple(output.type for output in init_function.outputs)
    else:
        transformed_function = function
        input_types = tuple(inp.type for inp in function.inputs)
    transformed_function = add_optimizer_ops(transformed_function)
    simulation = simulate(transformed_function, input_types, topology)
    latency = max([simulation.timestamps[d] for d in simulation.timestamps])
    peak_memory = max([simulation.peak_memory[d] for d in simulation.peak_memory])
    print(f"Latency: {latency} seconds")
    print(f"Throughput: {args.batch_size / latency:.2f} samples / second")
    print(f"Peak memory: {peak_memory / 1e9:.2f} GB")
    if args.trace_file is not None:
        simulation.dump_chrome_trace(args.trace_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP training and inference")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--input_dim", type=int, default=256, help="Input dim")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dim")
    parser.add_argument("--output_dim", type=int, default=256, help="Output dim")
    parser.add_argument(
        "--num_hidden_layers", type=int, default=16, help="# hidden layers"
    )
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
        "-k", "--num_microbatches", type=int, default=1, help="# of microbatches"
    )
    parser.add_argument(
        "--network_bandwidth", type=float, default=64, help="Network bandwidth in Gbps"
    )
    parser.add_argument(
        "--device_throughput", type=float, default=1.4e13, help="Device throughput"
    )
    parser.add_argument(
        "--dram_bandwidth", type=float, default=9e11, help="DRAM Bandwidth"
    )
    parser.add_argument(
        "--mode",
        choices=["training", "inference"],
        default="training",
        help="Execution mode",
    )
    parser.add_argument("--trace_file", type=str, default=None, help="Trace file")
    args = parser.parse_args()
    main(args)
