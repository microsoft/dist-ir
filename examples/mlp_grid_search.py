import argparse
import csv
from itertools import product
import numpy as np
import pandas as pd
import torch
from tqdm.contrib.concurrent import process_map
import os
import pickle

from dist_ir.ir import Topology
from dist_ir.executor import (
    infer_types,
    SequentialExecutor,
    Simulator,
    calibrate_device_parameters,
    calibrate_network_bandwidth,
    calibrate_allreduce_parameters,
)
from dist_ir.executor.cost_model import CostModel
from dist_ir.transforms import mlp_dhp_transform
from .mlp import mlp, calibrate_parameters, get_topology, simulate, run_pytorch


MODEL_PARAMS = {
    "mlp-tiny": (8, 512),
    "mlp-xs": (8, 4096),
    "mlp-small": (16, 8192),
    "mlp-medium": (64, 16384),
    "mlp-large": (128, 32768),
}


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
        backend,
        topology,
        allreduce_parameters,
    ) = config
    num_hidden_layers, input_dim = MODEL_PARAMS[model_size]
    hidden_dim = input_dim
    output_dim = hidden_dim
    d0 = topology.devices[0]
    function = mlp(batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, d0)
    function = infer_types(function, function.inputs)
    init_function, transformed_function = mlp_dist(
        function,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
        topology,
    )
    input_types = tuple(inp.type for inp in transformed_function.inputs)
    if backend == "simulate":
        latency, peak_memory = simulate(
            transformed_function, input_types, topology, allreduce_parameters
        )
        throughput = batch_size / latency
    elif backend == "pytorch":
        try:
            latency = run_pytorch(transformed_function, input_types, use_gpu=True)
            throughput = batch_size / latency
            peak_memory = 0
        except Exception as e:
            import traceback

            traceback.print_exc()
            latency = -1
            peak_memory = -1
            throughput = -1
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


def gen_configurations(
    all_model_sizes,
    all_world_sizes,
    all_batch_sizes,
    backend,
    topology,
    allreduce_parameters,
):
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
                        max(1, max_num_microbatches_exp - 5), max_num_microbatches_exp
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
                    backend,
                    topology,
                    allreduce_parameters,
                )


def grid_search(
    all_model_sizes,
    all_world_sizes,
    all_batch_sizes,
    backend,
    topology,
    allreduce_parameters,
):
    configs = list(
        gen_configurations(
            all_model_sizes,
            all_world_sizes,
            all_batch_sizes,
            backend,
            topology,
            allreduce_parameters,
        )
    )

    if backend == "pytorch":
        results = process_map(run_experiment, configs, chunksize=1, max_workers=1)
    else:
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
                backend,
                topology,
                allreduce_parameters,
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

def main(args):
    model_size = "mlp-xs"
    all_world_sizes = [1, 2, 4]
    all_batch_sizes = [1024, 2048, 4096, 8192]
    calibrate_parameters(args)
    topology = get_topology(
        max(all_world_sizes),
        device_throughput=args.device_throughput,
        dram_bandwidth=args.dram_bandwidth,
        kernel_launch_overhead=args.kernel_launch_overhead,
        network_bandwidth=args.network_bandwidth,
    )
    grid_search(
        all_model_sizes=[model_size],  # ["mlp-small", "mlp-medium", "mlp-large"],
        all_world_sizes=all_world_sizes,
        all_batch_sizes=all_batch_sizes,
        backend=args.backend,
        topology=topology,
        allreduce_parameters=args.allreduce_parameters,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["simulate", "pytorch"], required=True)
    parser.add_argument(
        "--device_throughput", type=float, default=1.4e13, help="Device throughput"
    )
    parser.add_argument(
        "--dram_bandwidth", type=float, default=9e11, help="DRAM Bandwidth"
    )
    parser.add_argument(
        "--kernel_launch_overhead",
        type=float,
        default=1e-5,
        help="Kernel launch overhead",
    )
    parser.add_argument(
        "--network_bandwidth", type=float, default=64, help="Network bandwidth in Gbps"
    )
    parser.add_argument("--allreduce_parameters", default=None)
    parser.add_argument(
        "--calibrate_device_parameters", action="store_true", default=False
    )
    parser.add_argument(
        "--calibrate_network_bandwidth",
        action="store_true",
        default=False,
        help="Calibrate network bandwidth",
    )
    parser.add_argument(
        "--calibrate_allreduce_parameters", action="store_true", default=False
    )
    parser.add_argument(
        "--simulation_parameters_file",
        type=str,
        default=None,
        help="File to load/save simulation parameters from/to",
    )
    args = parser.parse_args()
    main(args)
