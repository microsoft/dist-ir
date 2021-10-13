import argparse
import json
import os
import pandas as pd

from dist_ir.executor import (
    calibrate_device_parameters,
    calibrate_network_bandwidth,
    calibrate_allreduce_parameters,
)


def calibrate_parameters(args):
    if args.calibrate_all:
        args.calibrate_device_parameters = True
        args.calibrate_network_bandwidth = True
        args.calibrate_allreduce_parameters = True
    if args.output_file is None:
        raise ValueError(
            "Output file must be specified to calibrate simulation parameters"
        )
    if os.path.exists(args.output_file):
        print(f"Reading simulation parameters from {args.output_file}...")
        with open(args.output_file, "r") as f:
            simulation_parameters = json.load(f)
        if "device_parameters" in simulation_parameters:
            device_parameters = simulation_parameters["device_parameters"]
        else:
            device_parameters = [
                1.0 / args.dram_bandwidth,
                1.0 / args.device_throughput,
                args.kernel_launch_overhead,
            ]
        if "network_bandwidth" in simulation_parameters:
            network_bandwidth = simulation_parameters["network_bandwidth"]
        else:
            network_bandwidth = args.network_bandwidth
        if "allreduce_parameters" in simulation_parameters:
            allreduce_parameters = simulation_parameters["allreduce_parameters"]
        else:
            allreduce_parameters = args.allreduce_parameters
    else:
        simulation_parameters = {}
    update_simulation_parameters = False
    if args.calibrate_device_parameters:
        print("Calibrating device parameters...")
        device_parameters = calibrate_device_parameters(args.dtype)
        update_simulation_parameters = True
        print(f"DRAM bandwidth: {1.0 / device_parameters[0]:.2e}")
        print(f"Device throughput: {1.0 / device_parameters[1]:.2e}")
        print(f"Kernel launch overhead: {device_parameters[2]:.2e}")
    if args.calibrate_network_bandwidth:
        network_bandwidth = calibrate_network_bandwidth(args.dtype)
        update_simulation_parameters = True
        print(f"Network bandwidth: {network_bandwidth}")
    if args.calibrate_allreduce_parameters:
        allreduce_parameters = calibrate_allreduce_parameters(args.dtype)
        update_simulation_parameters = True
        print(f"Allreduce parameters: {allreduce_parameters}")
    if update_simulation_parameters:
        simulation_parameters["device_parameters"] = device_parameters
        simulation_parameters["network_bandwidth"] = network_bandwidth
        simulation_parameters["allreduce_parameters"] = allreduce_parameters
        with open(args.output_file, "w") as f:
            json.dump(simulation_parameters, f)


def prepare_best_grid_search_configs(args):
    if args.simulation_file is None:
        raise ValueError("Simulation file must be provided")
    # TODO handle files containing multiple model(-size)s
    best_configs = None
    df = pd.read_csv(args.simulation_file)
    model_sizes = df["model_size"].unique()
    batch_sizes = sorted(df["batch_size"].unique())
    if "mlp" in model_sizes[0]:
        model_sizes = ["mlp-small", "mlp-medium", "mlp-large"]
    elif "gpt" in model_sizes[0]:
        model_sizes = ["gpt3-xl", "gpt3-13B", "gpt3-175B"]
    for batch_size in batch_sizes:
        for model_size in model_sizes:
            df = pd.read_csv(args.simulation_file)
            df = df[(df["model_size"] == model_size) & (df["batch_size"] == batch_size)]
            df = df[df["peak_memory"] < 28 * 1e9]  # TODO make memory limit an arg
            df = df.sort_values(by="throughput", ascending=False).head(10)
            if best_configs is None:
                best_configs = df
            else:
                best_configs = best_configs.append(df)
    best_configs.to_csv(args.output_file)


def prepare_accuracy_sample_configs(args):
    if args.simulation_file is None:
        raise ValueError("Simulation file must be provided")
    df = pd.read_csv(args.simulation_file)
    model_sizes = df["model_size"].unique()
    if "mlp" in model_sizes[0]:
        model_sizes = ["mlp-small", "mlp-medium", "mlp-large"]
    elif "gpt" in model_sizes[0]:
        model_sizes = ["gpt3-xl", "gpt3-13B", "gpt3-175B"]
    sample_configs = None
    for model_size in model_sizes:
        df = pd.read_csv(args.simulation_file)
        df = df[df["model_size"] == model_size]
        df = df[df["peak_memory"] < 28 * 1e9]  # TODO make memory limit an arg
        df = df.sample(n=min(100, len(df)), random_state=args.seed)
        df = df.sort_values(by="peak_memory")
        if sample_configs is None:
            sample_configs = df
        else:
            sample_configs = sample_configs.append(df)
    sample_configs.to_csv(args.output_file)


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
    parser.add_argument(
        "--calibrate_all",
        action="store_true",
        default=False,
        help="Calibrate all parameters",
    )
    parser.add_argument(
        "--dtype", choices=["fp32", "fp16"], default="fp16", help="dtype"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    args = parser.parse_args()
    assert args.mode is not None

    if args.mode == "calibrate":
        calibrate_parameters(args)
    if args.mode == "prep-best":
        prepare_best_grid_search_configs(args)
    elif args.mode == "prep-sample":
        prepare_accuracy_sample_configs(args)
