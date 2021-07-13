import csv
from itertools import product
import numpy as np
from tqdm.contrib.concurrent import process_map

from dist_ir.ir import Topology
from dist_ir.executor import infer_types, Simulator
from dist_ir.executor.cost_model import CostModel
from dist_ir.transforms import mlp_dhp_transform
from .mlp import mlp

DGX_BANDWIDTH_GBPS = 200


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
    try:
        (
            batch_size,
            input_dim,
            num_hidden_layers,
            dp_degree,
            hp_degree,
            pp_degree,
            num_microbatches,
        ) = config
        hidden_dim = input_dim
        output_dim = hidden_dim
        topology = Topology()
        d0 = topology.add_device("gpu")
        function = mlp(
            batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, d0
        )
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
        simulation = simulator.interpret(
            transformed_function,
            (v.type for v in transformed_function.inputs),
        )
        latency = max([simulation.timestamps[d] for d in simulation.timestamps])
        throughput = batch_size / latency
        peak_memory = max([simulation.peak_memory[d] for d in simulation.timestamps])
        return latency, throughput, peak_memory
    except Exception as e:
        import sys, traceback

        traceback.print_exc()
        sys.exit(1)


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


def gen_configurations(hidden_dims, cluster_sizes, all_num_layers, all_batch_sizes):
    for hidden_dim, num_hidden_layers, batch_size, cluster_size in product(
        hidden_dims, all_num_layers, all_batch_sizes, cluster_sizes
    ):
        all_degrees = get_all_degrees(cluster_size)
        for (dp_degree, hp_degree, pp_degree) in all_degrees:
            if num_hidden_layers % pp_degree != 0:
                continue
            dp_batch_size = batch_size // dp_degree
            if pp_degree == 1:
                all_num_microbatches = [1]
            else:
                all_num_microbatches = [
                    int(2 ** k)
                    for k in range(1, int(np.floor(np.log2(dp_batch_size) / 2)))
                ]
            for num_microbatches in all_num_microbatches:
                if pp_degree == 1:
                    num_microbatches == 1
                yield (
                    batch_size,
                    hidden_dim,
                    num_hidden_layers,
                    dp_degree,
                    hp_degree,
                    pp_degree,
                    num_microbatches,
                )


def grid_search(hidden_dims, cluster_sizes, all_num_layers, all_batch_sizes):
    configs = list(
        gen_configurations(hidden_dims, cluster_sizes, all_num_layers, all_batch_sizes)
    )

    results = process_map(run_experiment, configs)

    with open("mlp_grid_search_results.csv", "w", newline="") as f:
        fieldnames = [
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
        for config, latency, throughput, peak_memory in zip(configs, results):
            (
                batch_size,
                input_dim,
                num_hidden_layers,
                dp_degree,
                hp_degree,
                pp_degree,
                num_microbatches,
            ) = config
            writer.writerow(
                {
                    "dp_degree": dp_degree,
                    "hp_degree": hp_degree,
                    "pp_degree": pp_degree,
                    "num_microbatches": num_microbatches,
                    "latency": latency,
                    "throughput": throughput,
                    "peak_memory": peak_memory,
                }
            )


if __name__ == "__main__":
    grid_search(
        hidden_dims=[8192, 32768],
        cluster_sizes=[16, 64],
        all_num_layers=[64],
        all_batch_sizes=[2048, 8192],
    )
