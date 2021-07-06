import argparse
import csv
import itertools
import numpy as np
import multiprocessing
import os
import tqdm

from dist_ir.importer import import_from_onnx
from dist_ir.ir import FunctionMaker, cpprint, Device, Topology, Value
from dist_ir.ir.type import Float32, Tensor
from dist_ir.executor import (
    CostModel,
    SequentialExecutor,
    PostTypeInferenceSimulator,
)
from dist_ir.transforms import gpt2_dhp_transform, filter_transform
from . import gpt2

MODEL_PARAMS = {
    "gpt2": (12, 12, 768),
    "gpt2-medium": (24, 16, 1024),
    "gpt2-large": (36, 20, 1280),
    "gpt2-xl": (48, 25, 1600),
    "gpt2-xl": (48, 25, 1600),
    "gpt3": (12, 12, 768),
    "gpt3-medium": (24, 16, 1024),
    "gpt3-large": (24, 16, 1536),
    "gpt3-xl": (24, 16, 2048),
    "gpt3-2.7B": (32, 32, 2560),
    "gpt3-6.7B": (32, 32, 4096),
    "gpt3-13B": (40, 40, 5120),
}


def get_all_degrees(n):
    """Given power-of-two world size n, returns all power-of-two factorizations of n."""
    if int(np.log2(n)) != np.log2(n):
        raise ValueError("World size must be a power of two")
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


def run(config):
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
        backend,
    ) = config
    n_layer, n_head, n_embd = MODEL_PARAMS[model_size]
    try:
        (
            transformed_function,
            initialized_input_data,
            topology,
        ) = gpt2.get_transformed_function_and_input_data(
            model_path,
            device_throughput,
            dram_bandwidth,
            network_bandwidth,
            batch_size,
            dp_degree,
            hp_degree,
            pp_degree,
            num_microbatches,
            n_layer,
            n_head,
            n_embd,
        )
        if backend == "simulate":
            simulation = gpt2.simulate(
                transformed_function, initialized_input_data, topology
            )
            latency = max([simulation.timestamps[d] for d in simulation.timestamps])
            peak_memory = max(
                [simulation.peak_memory[d] for d in simulation.peak_memory]
            ) / (2.0 ** 20)
        elif backend == "pytorch":
            world_size = len(topology.devices) - 1
            per_rank_outputs, runtimes = gpt2.run_pytorch(
                transformed_function, initialized_input_data, world_size
            )
            latency = np.median(runtimes[-1])
            # TODO: Measure peak memory?
            peak_memory = 0
    except Exception as e:
        latency = -1
        peak_memory = -1

    condensed_config = (
        model_size,
        batch_size,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
        latency,
        peak_memory,
    )

    return condensed_config, latency, peak_memory


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
    all_cluster_sizes = [4, 8, 16]
    all_batch_sizes = [64, 256]
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
    if args.pytorch:
        backend = "pytorch"
    else:
        backend = "simulate"

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
                        backend,
                    )
                )
    with open(args.output_file, "w", newline="") as f:
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
            _, latency, peak_memory = func(config)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 Grid Search")
    parser.add_argument(
        "--pytorch", action="store_true", default=False, help="Use PyTorch backend"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to GPT-2 ONNX model downloaded from "
            "https://github.com/onnx/models/blob/master/text/machine_comprehension/"
            "gpt-2/model/gpt2-10.onnx"
        ),
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
