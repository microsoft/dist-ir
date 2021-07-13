import argparse
import copy
import csv
import itertools
from multiprocessing import Manager
import numpy as np
import os
from tqdm.contrib.concurrent import process_map

from . import gpt2
from dist_ir.transforms import check_params

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

FIELDNAMES = [
    "model_size",
    "world_size",
    "batch_size",
    "dp_degree",
    "hp_degree",
    "pp_degree",
    "num_microbatches",
    "latency",
    "peak_memory",
]


def _get_all_degrees(n):
    """Given power-of-two world size n, returns all power-of-two factorizations of n."""
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


def _write_row(config, latency, peak_memory):
    (
        function,
        input_data,
        topology,
        output_file,
        model_size,
        world_size,
        batch_size,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
        backend,
        lock,
    ) = config
    with lock:
        with open(output_file, "a+", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(
                {
                    "model_size": model_size,
                    "world_size": world_size,
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


def run(config):
    (
        function,
        input_data,
        topology,
        output_file,
        model_size,
        world_size,
        batch_size,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
        backend,
        lock,
    ) = config
    n_layer, n_head, d_embd = MODEL_PARAMS[model_size]
    input_data = copy.deepcopy(input_data)
    try:
        init_function, transformed_function, initialized_input_data = gpt2.transform(
            function,
            input_data,
            topology,
            dp_degree,
            hp_degree,
            pp_degree,
            num_microbatches,
            d_embd,
            n_head,
            use_real_weights=(backend == "pytorch"),
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
    _write_row(config, latency, peak_memory)


def grid_search(args):
    if args.pytorch:
        raise NotImplementedError("Only grid search with simulation supported for now")
    # TODO: Make search space configuration part of args
    if os.path.exists(args.output_file):
        if (
            input(f'File "{args.output_file}" already exists. Overwrite? [y/n] ')
            .lower()
            .strip()[0]
            != "y"
        ):
            return
    all_world_sizes = [4, 8, 16]
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

    topology = gpt2.get_topology(
        max(all_world_sizes),
        args.device_throughput,
        args.dram_bandwidth,
        args.network_bandwidth,
    )
    base_model, base_input_data = gpt2.import_function_and_get_input_data(
        args.model_path, topology.devices[0]
    )
    models_and_input_data = {}
    for model_size in all_model_sizes:
        n_layer, n_head, d_embd = MODEL_PARAMS[model_size]
        models_and_input_data[model_size] = gpt2.resize_function_and_input_data(
            base_model,
            copy.deepcopy(base_input_data),
            n_layer,
            n_head,
            d_embd,
        )
    all_input_ids = gpt2.create_input_ids(max(all_batch_sizes))

    if args.pytorch:
        backend = "pytorch"
    else:
        backend = "simulate"

    manager = Manager()
    lock = manager.Lock()

    configs = []
    for model_size, world_size, batch_size in itertools.product(
        all_model_sizes, all_world_sizes, all_batch_sizes
    ):
        n_layer, n_head, d_embd = MODEL_PARAMS[model_size]
        model, input_data = models_and_input_data[model_size]
        input_ids = all_input_ids[:batch_size]
        input_data = [input_ids] + input_data
        all_degrees = _get_all_degrees(world_size)
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
                try:
                    check_params(
                        batch_size,
                        dp_degree,
                        hp_degree,
                        pp_degree,
                        num_microbatches,
                        n_head,
                        d_embd,
                    )
                    configs.append(
                        (
                            model,
                            input_data,
                            topology,
                            args.output_file,
                            model_size,
                            world_size,
                            batch_size,
                            dp_degree,
                            hp_degree,
                            pp_degree,
                            num_microbatches,
                            backend,
                            lock,
                        )
                    )
                except Exception as e:
                    print(
                        f"Skipping configuration dp_degree={dp_degree}, "
                        f"hp_degree={hp_degree}, pp_degree={pp_degree}, "
                        f"num_microbatches={num_microbatches}, "
                        f"n_head={n_head}, d_embd={d_embd}"
                    )
    # TODO: Use Pandas to manage output
    with open(args.output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
    process_map(run, configs)


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
            "Path to GPT-2 ONNX model "
            "(downloaded from https://github.com/onnx/models/blob/master/"
            "text/machine_comprehension/gpt-2/model/gpt2-10.onnx?raw=True)"
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
