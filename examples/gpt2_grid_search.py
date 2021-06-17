import argparse
from collections import defaultdict, OrderedDict
import csv
import itertools
import logging
import math
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing
import os
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
from . import gpt2

"""
model_params = {
    "gpt2": (12, 12, 768),
    "gpt2-medium": (24, 16, 1024),
    "gpt2-large": (36, 20, 1280),
    "gpt2-xl": (48, 25, 1600),
    "gpt2-xl": (48, 25, 1600),
}
"""
MODEL_PARAMS = {
    "gpt3": (12, 12, 768),
    "gpt3-medium": (24, 16, 1024),
    "gpt3-large": (24, 16, 1536),
    "gpt3-xl": (24, 24, 2048),
    "gpt3-2.7B": (32, 32, 2560),
    "gpt3-6.7B": (32, 32, 4096),
    "gpt3-13B": (40, 40, 5140),
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


def simulate(config):
    (
        model_path,
        device_throughput,
        dram_bandwidth,
        network_bandwidth,
        model_size,
        batch_size,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
    ) = config
    n_layer, n_head, n_embd = MODEL_PARAMS[model_size]
    topology = Topology()
    d0 = topology.add_device(
        "gpu", throughput=device_throughput, dram_bandwidth=dram_bandwidth
    )
    function, input_data = gpt2.import_function_and_get_input_data(
        model_path,
        batch_size=batch_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        default_device=d0,
    )
    ex = SequentialExecutor("numpy")
    function = ex.infer_types(
        function,
        input_data,
        input_devices=[topology.devices[0] for _ in range(len(input_data))],
    )
    condensed_config = (batch_size, dp_degree, hp_degree, pp_degree, num_microbatches)
    try:
        init_function, transformed_function, initialized_input_data = gpt2.transform(
            function,
            input_data,
            topology,
            dp_degree,
            hp_degree,
            pp_degree,
            num_microbatches,
            n_embd,
            device_throughput=device_throughput,
            dram_bandwidth=dram_bandwidth,
            network_bandwidth=network_bandwidth,
        )
        simulation = gpt2.simulate(
            transformed_function, initialized_input_data, topology
        )
        latency = max([simulation.timestamps[d] for d in simulation.timestamps])
        peak_memory = max(
            [simulation.peak_memory[d] for d in simulation.peak_memory]
        ) / (2.0 ** 20)
    except Exception as e:
        latency = -1
        peak_memory = -1
    return condensed_config, latency, peak_memory


def run_pytorch(config):
    (
        model_path,
        _,
        _,
        _,
        batch_size,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
    ) = config
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
    condensed_config = (batch_size, dp_degree, hp_degree, pp_degree, num_microbatches)
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
    except Exception as e:
        return condensed_config, -1, -1
    per_rank_outputs, runtimes = gpt2.run_pytorch(
        transformed_function, initialized_input_data, world_size
    )
    throughput = batch_size / np.median(runtimes[-1])
    # TODO: Measure peak memory?
    peak_memory = 0
    return condensed_config, throughput, peak_memory


def grid_search(args):
    # TODO: Make search space configuration part of args
    if os.path.exists(args.output_file):
        if (
            input(f'File "{args.output_file}" already exists. Overwrite? [y/n] ')
            .lower()
            .strip()[0]
            != "y"
        ):
            return
    all_cluster_sizes = [8]
    all_batch_sizes = [1, 4, 64, 256]
    # all_model_sizes = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    all_model_sizes = [
        "gpt3",
        "gpt3-medium",
        "gpt3-large",
        "gpt3-xl",
        "gpt3-2.7B",
        "gpt3-6.7B",
        "gpt3-13B",
    ]
    configs = []
    for model_size, cluster_size, batch_size in itertools.product(
        all_model_sizes, all_cluster_sizes, all_batch_sizes
    ):
        all_degrees = get_all_degrees(cluster_size)
        for (dp_degree, hp_degree, pp_degree) in all_degrees:
            if dp_degree > batch_size:
                continue
            elif pp_degree == 1:
                all_num_microbatches = [1]
            else:
                all_num_microbatches = [
                    int(2 ** k)
                    for k in range(
                        1,
                        int(
                            np.floor(
                                np.log2(batch_size // dp_degree) / 2,
                            )
                        ),
                    )
                ]
            for num_microbatches in all_num_microbatches:
                configs.append(
                    (
                        args.model_path,
                        args.device_throughput,
                        args.dram_bandwidth,
                        args.network_bandwidth,
                        model_size,
                        batch_size,
                        dp_degree,
                        hp_degree,
                        pp_degree,
                        num_microbatches,
                    )
                )
    with open("gpt2_grid_search_results.csv", "w", newline="") as f:
        fieldnames = [
            "model_size",
            "batch_size",
            "dp_degree",
            "hp_degree",
            "pp_degree",
            "num_microbatches",
            "latency",
            "peak_memory",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for config in tqdm.tqdm(configs):
            _, latency, peak_memory = simulate(config)
            (
                _,
                _,
                _,
                _,
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
                    "batch_size": batch_size,
                    "dp_degree": dp_degree,
                    "hp_degree": hp_degree,
                    "pp_degree": pp_degree,
                    "num_microbatches": num_microbatches,
                    "latency": latency,
                    "peak_memory": peak_memory,
                }
            )
            f.flush()

    """
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
                        for k in range(
                            1,
                            int(
                                np.floor(
                                    min(
                                        np.log2(pp_degree) + 1,
                                        np.log2(dp_batch_size) / 2,
                                    )
                                )
                            ),
                        )
                    ]
                for num_microbatches in all_num_microbatches:
                    if pp_degree == 1:
                        assert num_microbatches == 1
                    else:
                        assert num_microbatches > 1
                    configs.append(
                        (
                            args.model_path,
                            args.device_throughput,
                            args.dram_bandwidth,
                            args.network_bandwidth,
                            batch_size,
                            dp_degree,
                            hp_degree,
                            pp_degree,
                            num_microbatches,
                        )
                    )
    for config in configs:
        print(config)
    if not args.pytorch:
        n = multiprocessing.cpu_count()
        with multiprocessing.Pool(n) as pool:
            results = list(
                tqdm.tqdm(pool.imap_unordered(simulate, configs), total=len(configs))
            )
    else:
        results = []
        for config in tqdm.tqdm(configs):
            results.append(run_pytorch(config))

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
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 Grid Search")
    parser.add_argument(
        "--pytorch", action="store_true", default=False, help="Use PyTorch backend"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to GPT-2 ONNX model"
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
        "--output_file",
        type=str,
        default="gpt2_grid_search_results.csv",
        help="Output file",
    )
    args = parser.parse_args()
    grid_search(args)
