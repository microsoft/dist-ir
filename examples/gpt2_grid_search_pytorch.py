import numpy as np
import pandas as pd

from examples import gpt2

def main():
    df = pd.read_csv("gpt2_grid_search_results.csv")
    df = df.sort_values(by=["throughput", "latency"], ascending=[False, True])
    df = df[df["peak_memory"] * (2 ** 20) <= 12e9]
    print(df)
    keys = ["batch_size", "dp_degree", "hp_degree", "pp_degree", "num_microbatches"]
    model_path = "gpt2-10.onnx"
    device_throughput = 1.33e13
    dram_bandwidth = 6.58e11
    network_bandwidth = 8
    n_layer = 32
    n_head = 32
    d_embd = 4096
    results = []
    for (batch_size, dp_degree, hp_degree, pp_degree, num_microbatches) in df[keys].values[:10]:
        print(f"Running {batch_size}/{dp_degree}/{hp_degree}/{pp_degree}/{num_microbatches}...")
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
            d_embd,
            use_real_weights=True,
            print_stats=False,
        )
        world_size = dp_degree * hp_degree * pp_degree
        try:
            _, runtimes = gpt2.run_pytorch(
                transformed_function, initialized_input_data, world_size, use_gpu=True, debug_stacktrace=False
            )
            latency = np.median(runtimes[-1])
            throughput = batch_size / latency
            print(f"latency={latency*1000:.2f}, throughput={throughput:.2f}") 
        except RuntimeError as e:
            latency = np.inf
            throughput = -1
        results.append((batch_size, dp_degree, hp_degree, pp_degree, num_microbatches, latency, throughput))

    df = pd.DataFrame(results, columns=keys + ["latency", "throughput"])
    df.to_csv("gpt2_grid_search_results_pytorch.csv")


if __name__=='__main__':
    main()
