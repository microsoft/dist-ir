import argparse
from collections import defaultdict, OrderedDict
import csv
import logging
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing
from transformers import GPT2Tokenizer
import torch
import tqdm

import dist_ir
from dist_ir.importer import import_from_onnx
from dist_ir.ir import FunctionMaker, cpprint, pformat, Device, Topology, Value
from dist_ir.ir.type import Float32, Tensor
from dist_ir.executor import (
    CostModel,
    SequentialExecutor,
    PostTypeInferenceSimulator,
)
from dist_ir.transforms import gpt2_dhp_transform, filter_transform
import gpt2

NETWORK_BANDWIDTH_Gbps = 200
MODEL_PATH = "/lfs/1/keshav2/gpt2/model.onnx"


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


def simulate(config):
    (batch_size, dp_degree, hp_degree, pp_degree, num_microbatches) = config
    topology = Topology()
    d0 = topology.add_device("gpu")
    function, input_data = gpt2.import_function_and_get_input_data(
        MODEL_PATH, batch_size=batch_size, default_device=d0
    )
    ex = SequentialExecutor("numpy")
    function = ex.infer_types(
        function,
        input_data,
        input_devices=[topology.devices[0] for _ in range(len(input_data))],
    )
    try:
        init_function, transformed_function, initialized_input_data = gpt2.transform(
            function,
            input_data,
            topology,
            dp_degree,
            hp_degree,
            pp_degree,
            num_microbatches,
        )
        simulation = gpt2.simulate(transformed_function, initialized_input_data)
        throughput = batch_size / max(
            [simulation.timestamps[d] for d in simulation.timestamps]
        )
        peak_memory = max(
            [simulation.peak_memory[d] for d in simulation.peak_memory]
        ) / (2.0 ** 20)
    except Exception as e:
        throughput = 0
        peak_memory = 0
    return config, throughput, peak_memory


def run_pytorch(config):
    (batch_size, dp_degree, hp_degree, pp_degree, num_microbatches) = config
    world_size = dp_degree * hp_degree * pp_degree
    topology = Topology()
    d0 = topology.add_device("gpu")
    function, input_data = gpt2.import_function_and_get_input_data(
        MODEL_PATH, batch_size=batch_size, default_device=d0
    )
    ex = SequentialExecutor("numpy")
    function = ex.infer_types(
        function,
        input_data,
        input_devices=[topology.devices[0] for _ in range(len(input_data))],
    )
    init_function, transformed_function, initialized_input_data = gpt2.transform(
        function,
        input_data,
        topology,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
    )
    per_rank_outputs, runtimes = gpt2.run_pytorch(
        transformed_function, initialized_input_data, world_size
    )
    throughput = batch_size / np.median(runtimes[-1])
    # TODO: Measure peak memory?
    peak_memory = 0
    return config_throughput, peak_memory


def grid_search(args):
    # TODO: Make search space configuration part of args
    all_cluster_sizes = [4]
    all_batch_sizes = [64]
    configs = []
    for batch_size in all_batch_sizes:
        for i, cluster_size in enumerate(all_cluster_sizes):
            all_degrees = get_all_degrees(cluster_size)
            for (dp_degree, hp_degree, pp_degree) in all_degrees:
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
                        assert num_microbatches == 1
                    else:
                        assert num_microbatches > 1
                    configs.append(
                        (
                            batch_size,
                            dp_degree,
                            hp_degree,
                            pp_degree,
                            num_microbatches,
                        )
                    )
    for config in configs:
        print(config)
    if args.backend == "simulation":
        n = multiprocessing.cpu_count()
        target = simulate
    elif args.backend == "pytorch":
        n = 1
        target = run_pytorch
    with multiprocessing.Pool(n) as pool:
        results = list(
            tqdm.tqdm(pool.imap_unordered(target, configs), total=len(configs))
        )

    with open("grid_search_results.csv", "w", newline="") as f:
        fieldnames = [
            "batch_size",
            "dp_degree",
            "hp_degree",
            "pp_degree",
            "num_microbatches",
            "throughput",
            "peak_memory",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (config, throughput, peak_memory) in results:
            (
                batch_size,
                dp_degree,
                hp_degree,
                pp_degree,
                num_microbatches,
            ) = config
            writer.writerow(
                {
                    "batch_size": batch_size,
                    "dp_degree": dp_degree,
                    "hp_degree": hp_degree,
                    "pp_degree": pp_degree,
                    "num_microbatches": num_microbatches,
                    "throughput": throughput,
                    "peak_memory": peak_memory,
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 Grid Search")
    parser.add_argument(
        "--backend", choices=["simulation", "pytorch"], help="Simulation or PyTorch"
    )
    args = parser.parse_args()
    grid_search(args)
