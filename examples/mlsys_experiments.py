import argparse
import pandas as pd


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


def prepare_best_grid_search_configs(args):
    # TODO handle files containing multiple model(-size)s
    df = pd.read_csv(args.simulation_file)
    best_configs = df[df["peak_memory"] < 21e3]  # TODO make memory limit an arg
    best_configs = best_configs.sort_values(by="throughput", ascending=False).head(10)
    best_configs.to_csv(args.output_file)


def prepare_accuracy_sample_configs(args):
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
        required=True,
        help="File containing results of simulated grid search",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file",
    )
    args = parser.parse_args()
    assert args.mode is not None

    if args.mode == "prep-best":
        prepare_best_grid_search_configs(args)
    elif args.mode == "prep-sample":
        prepare_accuracy_sample_configs(args)
