import argparse
import csv
import itertools
import numpy as np
import time
import tqdm
import traceback
import torch

from dist_ir.ir import cpprint
from dist_ir.backend.torch import run_pytorch
from dist_ir.executor import (
    CostModel,
    Simulator,
    SequentialExecutor,
    calibrate_simulator,
    infer_types,
)
from dist_ir.transforms import mlp_dhp_transform
from examples import mlp

torch.manual_seed(42)


def get_inputs(batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers):
    x = torch.randn(size=(batch_size, input_dim), dtype=torch.float32)
    z = torch.randn(size=(batch_size, output_dim), dtype=torch.float32)
    weights = [torch.randn(size=(input_dim, hidden_dim), dtype=torch.float32)]
    for i in range(1, num_hidden_layers - 1):
        weights.append(torch.randn(size=(hidden_dim, hidden_dim), dtype=torch.float32))
    weights.append(torch.randn(size=(hidden_dim, output_dim), dtype=torch.float32))
    return x, z, weights


def mlp_dist_ir_simulation(
    batch_size,
    input_dim,
    hidden_dim,
    output_dim,
    num_hidden_layers,
    x,
    z,
    weights,
    device_throughput,
    dram_bandwidth,
    kernel_launch_overhead,
    max_memory_gb=10,
    warmup_steps=5,
    active_steps=50,
):
    topology = mlp.get_topology(
        1,
        device_throughput=device_throughput,
        dram_bandwidth=dram_bandwidth,
        kernel_launch_overhead=kernel_launch_overhead,
    )
    fn = mlp.mlp(
        batch_size,
        input_dim,
        hidden_dim,
        output_dim,
        num_hidden_layers,
        device=topology.devices[0],
    )
    input_types = tuple(inp.type for inp in fn.inputs)
    simulator = Simulator(CostModel(topology))
    simulation = simulator.interpret(fn, input_types)
    simulated_time = max([simulation.timestamps[d] for d in simulation.timestamps])
    peak_memory = max([simulation.peak_memory[d] for d in simulation.peak_memory])
    return simulated_time, peak_memory


def mlp_dist_ir_pytorch_backend(
    batch_size,
    input_dim,
    hidden_dim,
    output_dim,
    num_hidden_layers,
    x,
    z,
    weights,
    warmup_steps=5,
    active_steps=50,
    profile=False,
):
    topology = mlp.get_topology(1)
    fn = mlp.mlp(
        batch_size,
        input_dim,
        hidden_dim,
        output_dim,
        num_hidden_layers,
        device=topology.devices[0],
    )
    seq_executor = SequentialExecutor("numpy")
    input_data = [x, z] + weights
    fn = infer_types(fn, fn.inputs)

    # Measure actual execution time
    per_rank_outputs, runtimes = run_pytorch(
        fn,
        input_data,
        use_gpu=True,
        num_repetitions=active_steps,
        num_warmup=warmup_steps,
        profile=profile,
    )
    # TODO or median of max?
    actual_time = max(np.median(times) for times in runtimes)

    gradients = [
        per_rank_outputs[0][i] for i, v in enumerate(fn.outputs) if "dw" in v.name
    ]

    return gradients, actual_time


def mlp_pure_pytorch(x, z, weights, warmup_steps=5, active_steps=50, profile=False):
    batch_size = x.shape[0]
    x = x.cuda()
    z = z.cuda()
    weights = [w.cuda() for w in weights]
    events = []

    if active_steps < 10:
        print(
            "WARNING: The first active step includes large overhead, "
            "record more steps for a more accurate measurement"
        )

    def add_event():
        events.append(torch.cuda.Event(enable_timing=True))
        events[-1].record()

    if profile:
        wait_steps = 0
    else:
        wait_steps = warmup_steps + active_steps

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait_steps, warmup=warmup_steps, active=active_steps
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("mlp_pytorch_profile"),
    ) as p:
        for i in range(warmup_steps + active_steps):
            add_event()
            x_ = x.clone()
            z_ = z.clone()
            activations = [x_]
            matmul_outputs = []
            torch.cuda.empty_cache()
            add_event()
            for w_ in weights:
                x_ = torch.matmul(x_, w_)
                matmul_outputs.append(x_)
                x_ = torch.relu(x_)
                activations.append(x_)

            loss = torch.square(x_ - z_) / batch_size
            dy_ = 2 * (x_ - z_) / batch_size

            gradients = []
            for j, w_ in enumerate(reversed(weights)):
                x_ = matmul_outputs[len(matmul_outputs) - 1 - j]
                dy_[x_ <= 0] = 0
                a_ = activations[len(activations) - 2 - j]
                da_, dw_ = torch.matmul(dy_, w_.T), torch.matmul(a_.T, dy_)
                dy_ = da_
                gradients.append(dw_)
            if i == (warmup_steps + active_steps - 1):
                add_event()
            p.step()
        torch.cuda.synchronize()
        runtimes = [
            events[i].elapsed_time(events[i + 1]) / 1e3
            for i in range(1, len(events) - 1, 2)
        ]

    return gradients, np.median(runtimes[warmup_steps:])


def benchmark(
    batch_size,
    input_dim,
    hidden_dim,
    output_dim,
    num_hidden_layers,
    device_throughput,
    dram_bandwidth,
    kernel_launch_overhead,
    max_memory=10,
):
    x, z, weights = get_inputs(
        batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers
    )
    simulated_time, peak_memory = mlp_dist_ir_simulation(
        batch_size,
        input_dim,
        hidden_dim,
        output_dim,
        num_hidden_layers,
        x,
        z,
        weights,
        device_throughput,
        dram_bandwidth,
        kernel_launch_overhead,
    )
    if peak_memory / (1024 ** 3) > max_memory:
        return -1, -1, -1

    dist_ir_gradients, pytorch_backend_time = mlp_dist_ir_pytorch_backend(
        batch_size,
        input_dim,
        hidden_dim,
        output_dim,
        num_hidden_layers,
        x,
        z,
        weights,
    )

    torch.cuda.empty_cache()

    pytorch_gradients, pure_pytorch_time = mlp_pure_pytorch(x, z, weights)

    for x, y in zip(pytorch_gradients, dist_ir_gradients):
        np.testing.assert_array_almost_equal(
            x.detach().cpu().numpy(), y.detach().cpu().numpy(), decimal=2
        )

    return simulated_time, pytorch_backend_time, pure_pytorch_time


def grid_search(device_throughput, dram_bandwidth, kernel_launch_overhead):
    all_batch_sizes = [1024, 2048, 4096]
    all_dims = [1024, 2048, 4096]
    all_num_hidden_layers = [8, 12, 16]
    fieldnames = [
        "Batch size",
        "Dim",
        "Layers",
        "Simulated time",
        "PyTorch backend time",
        "Pure PyTorch time",
    ]

    with open("mlp_benchmark.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for (batch_size, dim, num_hidden_layers) in tqdm.tqdm(
            list(itertools.product(all_batch_sizes, all_dims, all_num_hidden_layers))
        ):
            try:
                simulated_time, pytorch_backend_time, pure_pytorch_time = benchmark(
                    batch_size,
                    dim,
                    dim,
                    dim,
                    num_hidden_layers,
                    device_throughput,
                    dram_bandwidth,
                    kernel_launch_overhead,
                )
            except Exception as e:
                traceback.print_exc()
                simulated_time = -1
                pytorch_backend_time = -1
                pure_pytorch_time = -1
            writer.writerow(
                [
                    batch_size,
                    dim,
                    num_hidden_layers,
                    simulated_time,
                    pytorch_backend_time,
                    pure_pytorch_time,
                ]
            )
            f.flush()
            torch.cuda.empty_cache()


def main(args):
    if args.calibrate and (args.mode == "simulate" or args.mode == "grid_search"):
        print("Calibrating simulator...")
        (
            args.dram_bandwidth,
            args.device_throughput,
            args.kernel_launch_overhead,
        ) = calibrate_simulator()
        print("Calibration results:")
        print(f"DRAM bandwidth: {args.dram_bandwidth:.2e}")
        print(f"Device throughput: {args.device_throughput:.2e}")
        print(f"Kernel launch overhead: {args.kernel_launch_overhead:.2e}")
    if args.mode == "grid_search":
        grid_search(
            args.device_throughput, args.dram_bandwidth, args.kernel_launch_overhead
        )
    elif args.mode == "simulate":
        x, z, weights = get_inputs(
            args.batch_size, args.dim, args.dim, args.dim, args.layers
        )
        simulated_time, peak_memory = mlp_dist_ir_simulation(
            args.batch_size,
            args.dim,
            args.dim,
            args.dim,
            args.layers,
            x,
            z,
            weights,
            args.device_throughput,
            args.dram_bandwidth,
            args.kernel_launch_overhead,
        )
        print(f"Simulated latency: {simulated_time * 1000:.2f} ms")
        print(f"Simulated peak memory: {peak_memory / (1024 ** 3):.2f} GB")
    elif args.mode == "backend":
        x, z, weights = get_inputs(
            args.batch_size, args.dim, args.dim, args.dim, args.layers
        )
        _, pytorch_backend_time = mlp_dist_ir_pytorch_backend(
            args.batch_size,
            args.dim,
            args.dim,
            args.dim,
            args.layers,
            x,
            z,
            weights,
            warmup_steps=args.warmup_steps,
            active_steps=args.active_steps,
            profile=args.profile,
        )
        print(f"PyTorch backend latency: {pytorch_backend_time * 1000:.2f} ms")
    elif args.mode == "pytorch":
        x, z, weights = get_inputs(
            args.batch_size, args.dim, args.dim, args.dim, args.layers
        )
        _, pure_pytorch_time = mlp_pure_pytorch(
            x,
            z,
            weights,
            warmup_steps=args.warmup_steps,
            active_steps=args.active_steps,
            profile=args.profile,
        )

        print(f"Pure PyTorch latency: {pure_pytorch_time * 1000:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP benchmark")
    parser.add_argument(
        "--mode",
        choices=["grid_search", "pytorch", "simulate", "backend"],
        default="simulation",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--dim", type=int, default=256, help="Weight dim")
    parser.add_argument("--layers", type=int, default=16, help="# layers")
    parser.add_argument("--warmup_steps", type=int, default=5, help="# warmup steps")
    parser.add_argument("--active_steps", type=int, default=100, help="# active steps")
    parser.add_argument(
        "--calibrate", action="store_true", default=False, help="Calibrate simulator"
    )
    parser.add_argument("--profile", action="store_true", default=False, help="Profile")
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
    args = parser.parse_args()
    main(args)
