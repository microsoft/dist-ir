import math
from pathlib import Path
import pandas as pd
import pytest
import tempfile
import torch

from dist_ir.utils import constants
from examples.grid_search import GridSearch
from examples.mlp_grid_search import MLPGridSearch
from examples.gpt2_grid_search import GPTGridSearch
from examples import mlp, gpt2

# Assume the onnx file is stored in the repository root
GPT2_MODEL_PATH = (Path(__file__).parent.parent / "gpt2-10.onnx").absolute()


@pytest.mark.parametrize(
    ("backend"),
    ["simulate", "pytorch"],
)
def test_mlp_grid_search(backend):
    all_world_sizes = [1, 2, 4]
    all_batch_sizes = [256]
    all_model_sizes = ["mlp-xs"]
    with tempfile.NamedTemporaryFile() as tf:
        grid_search = MLPGridSearch(
            backend,
            torch.cuda.is_available(),
            tf.name,
            constants.DEFAULT_DEVICE_THROUGHPUT,
            constants.DEFAULT_DRAM_BANDWIDTH,
            constants.DEFAULT_KERNEL_LAUNCH_OVERHEAD,
            constants.DEFAULT_NETWORK_BANDWIDTH,
        )
        grid_search.grid_search(all_world_sizes, all_batch_sizes, all_model_sizes)

        df = pd.read_csv(tf.name)

        if backend == "simulate":
            all_degrees = GridSearch.get_all_degrees(all_world_sizes[-1])
            num_layers, dim = grid_search.get_model_params(all_model_sizes[-1])
            for (d, t, p) in all_degrees:
                world_size = d * t * p
                simulation = mlp.run_mlp(
                    mode="training",
                    backend="simulate",
                    use_gpu=False,
                    batch_size=all_batch_sizes[0],
                    input_dim=dim,
                    hidden_dim=dim,
                    output_dim=dim,
                    num_hidden_layers=num_layers,
                    dp_degree=d,
                    hp_degree=t,
                    pp_degree=p,
                    num_microbatches=p,
                    device_throughput=constants.DEFAULT_DEVICE_THROUGHPUT,
                    dram_bandwidth=constants.DEFAULT_DRAM_BANDWIDTH,
                    kernel_launch_overhead=constants.DEFAULT_KERNEL_LAUNCH_OVERHEAD,
                    network_bandwidth=constants.DEFAULT_NETWORK_BANDWIDTH,
                    trace_file=None,
                    verbose=False,
                )
                latency = simulation.get_latency()
                grid_search_latency = df[
                    (df["model_size"] == all_model_sizes[-1])
                    & (df["world_size"] == world_size)
                    & (df["dp_degree"] == d)
                    & (df["hp_degree"] == t)
                    & (df["pp_degree"] == p)
                    & (df["num_microbatches"] == p)
                ]["latency"].values[0]
                assert math.isclose(latency, grid_search_latency, abs_tol=10 ** -8)

    # TODO: Check correctness for PyTorch?


@pytest.mark.parametrize(
    ("backend"),
    ["simulate", "pytorch"],
)
def test_gpt_grid_search(backend):
    all_world_sizes = [1, 2, 4]
    all_batch_sizes = [256]
    all_model_sizes = ["gpt3"]
    with tempfile.NamedTemporaryFile() as tf:
        grid_search = GPTGridSearch(
            backend,
            torch.cuda.is_available(),
            tf.name,
            GPT2_MODEL_PATH,
            constants.DEFAULT_DEVICE_THROUGHPUT,
            constants.DEFAULT_DRAM_BANDWIDTH,
            constants.DEFAULT_KERNEL_LAUNCH_OVERHEAD,
            constants.DEFAULT_NETWORK_BANDWIDTH,
        )
        grid_search.grid_search(all_world_sizes, all_batch_sizes, all_model_sizes)

        df = pd.read_csv(tf.name)

        if backend == "simulate":
            all_degrees = GridSearch.get_all_degrees(all_world_sizes[-1])
            n_layer, n_head, d_embd = grid_search.get_model_params(all_model_sizes[-1])
            for (d, t, p) in all_degrees:
                world_size = d * t * p
                (
                    transformed_fn,
                    initialized_input_data,
                    topology,
                ) = gpt2.get_transformed_function_and_input_data(
                    model_path=GPT2_MODEL_PATH,
                    device_throughput=constants.DEFAULT_DEVICE_THROUGHPUT,
                    dram_bandwidth=constants.DEFAULT_DRAM_BANDWIDTH,
                    kernel_launch_overhead=constants.DEFAULT_KERNEL_LAUNCH_OVERHEAD,
                    network_bandwidth=constants.DEFAULT_NETWORK_BANDWIDTH,
                    batch_size=all_batch_sizes[0],
                    dp_degree=d,
                    hp_degree=t,
                    pp_degree=p,
                    num_microbatches=p,
                    n_layer=n_layer,
                    n_head=n_head,
                    d_embd=d_embd,
                    use_real_weights=False,
                    print_stats=False,
                )
                simulation = gpt2.simulate(
                    transformed_fn, initialized_input_data, topology
                )
                latency = simulation.get_latency()
                grid_search_latency = df[
                    (df["model_size"] == all_model_sizes[-1])
                    & (df["world_size"] == world_size)
                    & (df["dp_degree"] == d)
                    & (df["hp_degree"] == t)
                    & (df["pp_degree"] == p)
                    & (df["num_microbatches"] == p)
                ]["latency"].values[0]
                assert math.isclose(latency, grid_search_latency, abs_tol=10 ** -8)


if __name__ == "__main__":
    test_mlp_grid_search("simulate")
    test_gpt_grid_search("simulate")
