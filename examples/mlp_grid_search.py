import csv
from itertools import product
import numpy as np
import pandas as pd
import torch
from tqdm.contrib.concurrent import process_map

from dist_ir.backend.torch import run_pytorch
from dist_ir.ir import Topology
from dist_ir.executor import infer_types, SequentialExecutor, Simulator
from dist_ir.executor.cost_model import CostModel
from dist_ir.transforms import mlp_dhp_transform
from .mlp import mlp

DGX_BANDWIDTH_GBPS = 200

MODEL_PARAMS = {
    "mlp-xs": (8, 512),
    "mlp-small": (16, 8192),
    "mlp-medium": (64, 16384),
    "mlp-large": (128, 32768),
}


def add_devices_to_topology(topology, num_devices):
    for i in range(num_devices):
        topology.add_device("gpu")
    devices = topology.devices
    for i in range(0, len(devices)):
        for j in range(i + 1, len(devices)):
            topology.set_bandwidth(devices[i], devices[j], DGX_BANDWIDTH_GBPS)


def get_all_degrees(n):
    all_degrees = []
    d = 1
    h = 1
    p = 1
    while d <= n:
        h = 1
        p = 1
        if d * h * p == n:
            all_degrees.append((d, h, p))
            break
        while h <= n:
            p = 1
            if d * h * p == n:
                all_degrees.append((d, h, p))
                break
            while p <= n:
                if d * h * p == n:
                    all_degrees.append((d, h, p))
                    break
                p *= 2
            h *= 2
        d *= 2
    return all_degrees


def run_experiment(config):
    (
        model_size,
        batch_size,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
    ) = config
    num_hidden_layers, input_dim = MODEL_PARAMS[model_size]
    hidden_dim = input_dim
    output_dim = hidden_dim
    # TODO topology can be created once and shared for all configs
    topology = Topology()
    d0 = topology.add_device("gpu")
    function = mlp(batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, d0)
    function = infer_types(function, function.inputs)
    world_size = dp_degree * hp_degree * pp_degree
    add_devices_to_topology(topology, world_size)
    init_function, transformed_function = mlp_dist(
        function,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
        topology,
    )
    simulator = Simulator(CostModel(topology))
    simulation = simulator.simulate(
        transformed_function,
        (v.type for v in transformed_function.inputs),
    )
    latency = max([simulation.timestamps[d] for d in simulation.timestamps])
    throughput = batch_size / latency
    peak_memory = max([simulation.peak_memory[d] for d in simulation.timestamps])
    return latency, throughput, peak_memory


def mlp_dist(
    mlp_fn,
    dp_degree,
    hp_degree,
    pp_degree,
    num_microbatches,
    topology,
):
    init_function, transformed_function = mlp_dhp_transform(
        mlp_fn,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
        topology.devices,
    )
    init_function = infer_types(init_function, init_function.inputs)
    # init_function.outputs = transformed_function.inputs, so get types from there:
    transformed_function = infer_types(transformed_function, init_function.outputs)
    return init_function, transformed_function


def gen_configurations(all_model_sizes, all_world_sizes, all_batch_sizes):
    for (
        model_size,
        world_size,
        batch_size,
    ) in product(all_model_sizes, all_world_sizes, all_batch_sizes):
        all_degrees = get_all_degrees(world_size)
        num_hidden_layers, hidden_dim = MODEL_PARAMS[model_size]
        for (dp_degree, hp_degree, pp_degree) in all_degrees:
            if num_hidden_layers % pp_degree != 0:
                continue
            dp_batch_size = batch_size // dp_degree
            if dp_batch_size == 0:
                continue
            if pp_degree == 1:
                all_num_microbatches = [1]
            else:
                max_num_microbatches_exp = int(np.floor(np.log2(dp_batch_size) / 2))
                all_num_microbatches = [
                    int(2 ** k)
                    for k in range(
                        max(1, max_num_microbatches_exp - 3), max_num_microbatches_exp
                    )
                ]
            for num_microbatches in all_num_microbatches:
                if pp_degree == 1:
                    num_microbatches == 1
                yield (
                    model_size,
                    batch_size,
                    dp_degree,
                    hp_degree,
                    pp_degree,
                    num_microbatches,
                )


def grid_search(all_model_sizes, all_world_sizes, all_batch_sizes):
    configs = list(
        gen_configurations(all_model_sizes, all_world_sizes, all_batch_sizes)
    )

    results = process_map(run_experiment, configs, chunksize=1)

    with open("mlp_grid_search_results.csv", "w", newline="") as f:
        fieldnames = [
            "model_size",
            "world_size",
            "batch_size",
            "dp_degree",
            "hp_degree",
            "pp_degree",
            "num_microbatches",
            "latency",
            "throughput",
            "peak_memory",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for config, (latency, throughput, peak_memory) in zip(configs, results):
            (
                model_size,
                batch_size,
                dp_degree,
                hp_degree,
                pp_degree,
                num_microbatches,
            ) = config
            writer.writerow(
                {
                    "model_size": model_size,
                    "world_size": dp_degree * hp_degree * pp_degree,
                    "batch_size": batch_size,
                    "dp_degree": dp_degree,
                    "hp_degree": hp_degree,
                    "pp_degree": pp_degree,
                    "num_microbatches": num_microbatches,
                    "latency": latency,
                    "throughput": throughput,
                    "peak_memory": peak_memory,
                }
            )


def grid_search_pytorch(all_model_sizes, all_world_sizes, all_batch_sizes):
    configs = gen_configurations(all_model_sizes, all_world_sizes, all_batch_sizes)

    with open("mlp_pytorch.csv", "w", newline="") as f:
        fieldnames = [
            "model_size",
            "world_size",
            "batch_size",
            "dp_degree",
            "hp_degree",
            "pp_degree",
            "num_microbatches",
            "latency_pt",
            "throughput_pt",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            try:
                latency, throughput = run_backend(config)
            except RuntimeError as e:
                print(e)
                latency, throughput = -1.0, -1.0
            (
                model_size,
                batch_size,
                dp_degree,
                hp_degree,
                pp_degree,
                num_microbatches,
            ) = config
            writer.writerow(
                {
                    "model_size": model_size,
                    "world_size": dp_degree * hp_degree * pp_degree,
                    "batch_size": batch_size,
                    "dp_degree": dp_degree,
                    "hp_degree": hp_degree,
                    "pp_degree": pp_degree,
                    "num_microbatches": num_microbatches,
                    "latency_pt": latency,
                    "throughput_pt": throughput,
                }
            )
            f.flush()


def get_inputs(batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers):
    x = torch.randn(size=(batch_size, input_dim), dtype=torch.float32)
    z = torch.randn(size=(batch_size, output_dim), dtype=torch.float32)
    weights = [torch.randn(size=(input_dim, hidden_dim), dtype=torch.float32)]
    for i in range(1, num_hidden_layers - 1):
        weights.append(torch.randn(size=(hidden_dim, hidden_dim), dtype=torch.float32))
    weights.append(torch.randn(size=(hidden_dim, output_dim), dtype=torch.float32))
    return x, z, weights


def run_backend(config):
    """Run given config on pytorch backend."""
    print(f"Config: {config}")
    (
        model_size,
        batch_size,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
    ) = config
    num_hidden_layers, input_dim = MODEL_PARAMS[model_size]
    hidden_dim = input_dim
    output_dim = hidden_dim
    topology = Topology()
    d0 = topology.add_device("gpu")
    function = mlp(batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, d0)
    function = infer_types(function, function.inputs)
    world_size = dp_degree * hp_degree * pp_degree
    add_devices_to_topology(topology, world_size)
    init_function, transformed_function = mlp_dist(
        function,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
        topology,
    )
    x, z, weights = get_inputs(
        batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers
    )
    input_data = [x, z] + weights
    if world_size > 1:
        ex = SequentialExecutor("numpy")
        input_data = [
            torch.from_numpy(v).to(torch.float32)
            for v in ex.compute(init_function, [v.numpy() for v in input_data])
        ]

    # Measure actual execution time
    _, runtimes = run_pytorch(
        transformed_function,
        input_data,
        use_gpu=True,
        num_repetitions=10,
        num_warmup=5,
        profile=False,
    )
    # TODO or median of max?
    actual_time = max(np.median(times) for times in runtimes)
    throughput = batch_size / actual_time
    print(f"Runtime: {actual_time}\nThroughput: {throughput}")
    return actual_time, throughput


class MLP(torch.nn.Module):
    def __init__(self, weights):
        super(MLP, self).__init__()
        self.weights = [torch.nn.parameter.Parameter(w) for w in weights]

    def forward(self, x):
        for w in self.weights:
            # TODO add bias to our mlp and use nn.Linear here
            x = torch.matmul(x, w)
            x = torch.relu(x)
        return x
        # TODO confirm this gives same output as the equivalent DistIR mlp fn


def run_vanilla_baseline(model_size, batch_size):
    """Run sequential model on vanilla pytorch"""
    print(f"Config: {(batch_size, 1, 1, 1, 1)}")
    num_hidden_layers, input_dim = MODEL_PARAMS[model_size]
    hidden_dim = input_dim
    output_dim = hidden_dim
    events = []
    warmup_steps = 5
    active_steps = 10

    x, z, weights = get_inputs(
        batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers
    )
    x = x.cuda(0)
    z = z.cuda(0)  # loss needs integer z. Why is it float32 in DistIR?
    weights = [w.cuda(0) for w in weights]

    model = MLP(weights).cuda(0)
    loss = torch.nn.MSELoss()

    def add_event():
        events.append(torch.cuda.Event(enable_timing=True))
        events[-1].record()

    for _ in range(warmup_steps + active_steps):
        # TODO do I need to zero gradients here?
        add_event()
        y = model(x)
        l = loss(y, z)
        l.backward()
        # TODO we should add optimizer to DistIR model and here
        add_event()

    torch.cuda.synchronize()
    runtimes = [
        events[i].elapsed_time(events[i + 1]) / 1e3 for i in range(len(events) - 1)
    ]
    latency = np.median(runtimes[warmup_steps:])
    throughput = batch_size / latency
    print(f"Runtime: {latency}\nThroughput: {throughput}")
    return latency, throughput


if __name__ == "__main__":
    torch.manual_seed(42)
    model_size = "mlp-small"

    # # Grid search simulation to find best configuration:
    # grid_search(
    #     all_model_sizes=[model_size],  # ["mlp-small", "mlp-medium", "mlp-large"],
    #     all_world_sizes=[1, 2, 4],
    #     all_batch_sizes=[2 ** i for i in range(16)]
    #     # all_batch_sizes=[512, 1024, 2048, 4096, 8192],
    # )

    # # Run sequential baseline on pytorch backend
    # for i in range(10, 15):
    #     run_backend((model_size, 2 ** i, 1, 1, 1, 1))

    # Try pure DP/HP/PP baselines on pytorch backend:
    # # DP goes OOM even with BS=4
    # for i in range(1, 15):
    #     run_backend((model_size, 2 ** i, 4, 1, 1, 1))
    # # HP:
    # try:
    #     for i in range(12, 20):
    #         run_backend((model_size, 2 ** i, 1, 4, 1, 1))
    # except RuntimeError as e:
    #     print(e)
    # # PP:
    # try:
    #     for i in [6]:  # range(1, 20):
    #         run_backend((model_size, 16384, 1, 1, 4, 2 ** i))
    # except RuntimeError as e:
    #     print(e)
    #     # TODO does (2, 1, 1, 4, 2) have effective batch size 2 or 4?

    # # Run best configs on pytorch backend
    # df = pd.read_csv("mlp_grid_search_results.csv")
    # # Use a 8GB memory estimate cutoff to avoid OOMs as much as possible
    # # df = df[df["peak_memory"] < 14e9]
    # for _, row in df.sort_values(by="throughput", ascending=False).iterrows():
    #     config = (
    #         model_size,
    #         row["batch_size"],
    #         row["dp_degree"],
    #         row["hp_degree"],
    #         row["pp_degree"],
    #         row["num_microbatches"],
    #     )
    #     try:
    #         run_backend(config)
    #     except RuntimeError as e:
    #         print(e)

    # # Run sequential model on vanilla pytorch as baseline:
    # try:
    #     for i in range(10, 20):
    #         run_vanilla_baseline(model_size, 2 ** i)
    # except RuntimeError as e:
    #     print(e)

    # Grid search pytorch backend:
    grid_search_pytorch(
        all_model_sizes=[model_size],  # ["mlp-small", "mlp-medium", "mlp-large"],
        all_world_sizes=[1, 2, 4],
        all_batch_sizes=[2 ** i for i in range(16)],
    )
