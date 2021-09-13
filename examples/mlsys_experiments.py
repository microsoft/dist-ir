import argparse
import json
import os
import pandas as pd

from dist_ir.executor import (
    calibrate_device_parameters,
    calibrate_network_bandwidth,
    calibrate_allreduce_parameters,
)

"""
def mlp_training():
    # TODO: Get these from calibration
    device_throughput = constants.DEFAULT_DEVICE_THROUGHPUT
    dram_bandwidth = constants.DEFAULT_DRAM_BANDWIDTH
    kernel_launch_overhead = constants.DEFAULT_KERNEL_LAUNCH_OVERHEAD
    network_bandwidth = constants.DEFAULT_NETWORK_BANDWIDTH
    all_model_sizes = ["mlp-small"]
    all_world_sizes = [1, 2, 4]
    all_batch_sizes = [2 ** i for i in range(16)]

    # Grid search simulation to find best configuration:
    grid_search = MLPGridSearch(
        backend="simulate",
        use_gpu=False,
        output_file="mlsys_mlp_grid_search_results.csv",
        device_throughput=device_throughput,
        dram_bandwidth=dram_bandwidth,
        kernel_launch_overhead=kernel_launch_overhead,
        network_bandwidth=network_bandwidth,
    )
    grid_search.grid_search(all_world_sizes, all_batch_sizes, all_model_sizes)

    # TODO: Finish
    # Run sequential baseline on pytorch backend
    for i in range(10, 15):
        mlp.run_backend((model_size, 2 ** i, 1, 1, 1, 1))

    # Try pure DP/HP/PP baselines on pytorch backend:
    # DP goes OOM even with BS=4
    for i in range(1, 15):
        run_backend((model_size, 2 ** i, 4, 1, 1, 1))
    # HP:
    try:
        for i in range(12, 20):
            run_backend((model_size, 2 ** i, 1, 4, 1, 1))
    except RuntimeError as e:
        print(e)
    # PP:
    try:
        for i in [6]:  # range(1, 20):
            run_backend((model_size, 16384, 1, 1, 4, 2 ** i))
    except RuntimeError as e:
        print(e)
        # TODO does (2, 1, 1, 4, 2) have effective batch size 2 or 4?

    # Run best configs on pytorch backend
    df = pd.read_csv("mlp_grid_search_results.csv")
    # Use a 8GB memory estimate cutoff to avoid OOMs as much as possible
    # df = df[df["peak_memory"] < 14e9]
    for _, row in df.sort_values(by="throughput", ascending=False).iterrows():
        config = (
            model_size,
            row["batch_size"],
            row["dp_degree"],
            row["hp_degree"],
            row["pp_degree"],
            row["num_microbatches"],
        )
        try:
            run_backend(config)
        except RuntimeError as e:
            print(e)

    # Run sequential model on vanilla pytorch as baseline:
    try:
        for i in range(10, 20):
            run_vanilla_baseline(model_size, 2 ** i)
    except RuntimeError as e:
        print(e)
    """


def calibrate_parameters(args):
    if args.output_file is None:
        raise ValueError(
            "Output file must be specified to calibrate simulation parameters"
        )
    if os.path.exists(args.output_file):
        print(f"Reading simulation parameters from {args.output_file}...")
        with open(args.output_file, "r") as f:
            simulation_parameters = json.load(f)
        if "device_throughput" in simulation_parameters:
            device_throughput = simulation_parameters["device_throughput"]
        else:
            assert args.calibrate_device_parameters
        if "dram_bandwidth" in simulation_parameters:
            dram_bandwidth = simulation_parameters["dram_bandwidth"]
        else:
            assert args.calibrate_device_parameters
        if "kernel_launch_overhead" in simulation_parameters:
            kernel_launch_overhead = simulation_parameters["kernel_launch_overhead"]
        else:
            assert args.calibrate_device_parameters
        if "network_bandwidth" in simulation_parameters:
            network_bandwidth = simulation_parameters["network_bandwidth"]
        else:
            assert args.calibrate_network_bandwidth
        if "allreduce_parameters" in simulation_parameters:
            allreduce_parameters = simulation_parameters["allreduce_parameters"]
        else:
            assert args.calibrate_allreduce_parameters
    else:
        simulation_parameters = {}
    update_simulation_parameters = False
    if args.calibrate_device_parameters:
        print("Calibrating device parameters...")
        (
            dram_bandwidth,
            device_throughput,
            kernel_launch_overhead,
        ) = calibrate_device_parameters()
        update_simulation_parameters = True
        print(f"DRAM bandwidth: {dram_bandwidth:.2e}")
        print(f"Device throughput: {device_throughput:.2e}")
        print(f"Kernel launch overhead: {kernel_launch_overhead:.2e}")
    if args.calibrate_network_bandwidth:
        network_bandwidth = calibrate_network_bandwidth()
        update_simulation_parameters = True
        print(f"Network bandwidth: {network_bandwidth}")
    if args.calibrate_allreduce_parameters:
        allreduce_parameters = calibrate_allreduce_parameters()
        update_simulation_parameters = True
        print(f"Allreduce parameters: {allreduce_parameters}")
    if update_simulation_parameters:
        simulation_parameters["dram_bandwidth"] = dram_bandwidth
        simulation_parameters["device_throughput"] = device_throughput
        simulation_parameters["kernel_launch_overhead"] = kernel_launch_overhead
        simulation_parameters["network_bandwidth"] = network_bandwidth
        simulation_parameters["allreduce_parameters"] = allreduce_parameters
        with open(args.output_file, "w") as f:
            json.dump(simulation_parameters, f)


def prepare_best_grid_search_configs(args):
    if args.simulation_file is None:
        raise ValueError("Simulation file must be provided")
    # TODO handle files containing multiple model(-size)s
    df = pd.read_csv(args.simulation_file)
    best_configs = df[df["peak_memory"] < 21e3]  # TODO make memory limit an arg
    best_configs = best_configs.sort_values(by="throughput", ascending=False).head(10)
    best_configs.to_csv(args.output_file)


def prepare_accuracy_sample_configs(args):
    if args.simulation_file is None:
        raise ValueError("Simulation file must be provided")
    df = pd.read_csv(args.simulation_file)
    sample = df[df["peak_memory"] < 21e3]  # TODO make memory limit an arg
    sample = sample.sample(n=20, random_state=1)
    sample = sample.sort_values(by="peak_memory")
    sample.to_csv(args.output_file)


if __name__ == "__main__":
    # 0. Calibrate simulation params (TODO after merging simulator_accuracy)
    # 1. Simulation grid search (use X_grid_search scripts directly?)
    # 2. Get list of configs for backend
    # (3. Use bash script to run backend on configs)
    # 4. Use simulation & backend results to plot results, create tables
    # 5. TODO run against vanilla PyTorch models

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["calibrate", "prep-best", "prep-sample", "plot"],
        default=None,
        help="Run mode",
    )
    parser.add_argument(
        "--simulation_file",
        type=str,
        default=None,
        help="File containing results of simulated grid search",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file",
    )
    parser.add_argument(
        "--calibrate_device_parameters",
        action="store_true",
        default=False,
        help="Calibrate device parameters",
    )
    parser.add_argument(
        "--calibrate_network_bandwidth",
        action="store_true",
        default=False,
        help="Calibrate network bandwidth",
    )
    parser.add_argument(
        "--calibrate_allreduce_parameters",
        action="store_true",
        default=False,
        help="Calibrate allreduce parameters",
    )

    args = parser.parse_args()
    assert args.mode is not None

    if args.mode == "calibrate":
        calibrate_parameters(args)
    if args.mode == "prep-best":
        prepare_best_grid_search_configs(args)
    elif args.mode == "prep-sample":
        prepare_accuracy_sample_configs(args)
