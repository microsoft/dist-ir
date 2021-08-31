import argparse
import csv
import itertools
import numpy as np
import os
import pickle
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
    calibrate_device_parameters,
    calibrate_network_bandwidth,
    infer_types,
)
from dist_ir.transforms import mlp_dhp_transform
from examples import mlp, mlp_grid_search

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
    network_bandwidth,
    d,
    t,
    p,
    k,
    max_memory_GB=10,
    warmup_steps=5,
    active_steps=50,
    verbose=False,
):
    world_size = d * t * p
    topology = mlp.get_topology(
        world_size,
        device_throughput=device_throughput,
        dram_bandwidth=dram_bandwidth,
        kernel_launch_overhead=kernel_launch_overhead,
        network_bandwidth=network_bandwidth,
    )
    fn = mlp.mlp(
        batch_size,
        input_dim,
        hidden_dim,
        output_dim,
        num_hidden_layers,
        device=topology.devices[0],
    )
    if world_size > 1:
        init_fn, fn = mlp_dhp_transform(fn, d, t, p, k, topology.devices)
        init_fn = infer_types(init_fn, init_fn.inputs)
        input_types = tuple(output.type for output in init_fn.outputs)
    else:
        input_types = tuple(inp.type for inp in fn.inputs)
    if verbose:
        init_fn = infer_types(init_fn, init_fn.inputs)
        fn = infer_types(fn, init_fn.outputs)
        cpprint(fn)

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
    d,
    t,
    p,
    k,
    warmup_steps=5,
    active_steps=50,
    profile=False,
    verbose=False,
):
    world_size = d * t * p
    topology = mlp.get_topology(world_size)
    fn = mlp.mlp(
        batch_size,
        input_dim,
        hidden_dim,
        output_dim,
        num_hidden_layers,
        device=topology.devices[0],
    )
    input_data = [x, z] + weights
    if world_size > 1:
        init_fn, fn = mlp_dhp_transform(fn, d, t, p, k, topology.devices)
        init_fn = infer_types(init_fn, init_fn.inputs)
        fn = infer_types(fn, init_fn.outputs)
        ex = SequentialExecutor("numpy")
        input_data = [
            torch.from_numpy(v).to(torch.float32)
            for v in ex.compute(init_fn, [v.numpy() for v in input_data])
        ]
    if verbose:
        fn = infer_types(fn, fn.inputs)
        cpprint(fn)

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

    if world_size == 1:
        gradients = [
            per_rank_outputs[0][i] for i, v in enumerate(fn.outputs) if "dw" in v.name
        ]
    else:
        gradients = None

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
    network_bandwidth,
    d=1,
    t=1,
    p=1,
    k=1,
    max_memory_GB=10,
):
    world_size = d * t * p
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
        network_bandwidth,
        d,
        t,
        p,
        k,
    )
    return simulated_time, -1

    if peak_memory / (1024 ** 3) > max_memory_GB:
        if world_size == 1:
            return -1, -1, -1
        else:
            return -1, -1

    dist_ir_gradients, pytorch_backend_time = mlp_dist_ir_pytorch_backend(
        batch_size,
        input_dim,
        hidden_dim,
        output_dim,
        num_hidden_layers,
        x,
        z,
        weights,
        d,
        t,
        p,
        k,
    )
    torch.cuda.empty_cache()

    if world_size == 1:
        pytorch_gradients, pure_pytorch_time = mlp_pure_pytorch(x, z, weights)

        for x, y in zip(pytorch_gradients, dist_ir_gradients):
            np.testing.assert_array_almost_equal(
                x.detach().cpu().numpy(), y.detach().cpu().numpy(), decimal=2
            )

        return simulated_time, pytorch_backend_time, pure_pytorch_time
    else:
        return simulated_time, pytorch_backend_time


def distributed_grid_search(
    device_throughput, dram_bandwidth, kernel_launch_overhead, network_bandwidth
):
    batch_size = 8192
    all_dims = [1024, 2048, 4096]
    all_num_layers = [8, 16]
    world_size = 8  # torch.cuda.device_count()
    all_degrees = mlp_grid_search.get_all_degrees(world_size)
    configs = []
    for (dim, num_layers) in itertools.product(all_dims, all_num_layers):
        for (d, t, p) in all_degrees:
            if p == 1:
                k = 1
                configs.append((d, t, p, k, dim, num_layers))
            else:
                for i in range(1, 5):
                    k = int(2 ** i)
                    configs.append((d, t, p, k, dim, num_layers))

    fieldnames = [
        "Dim",
        "Layers",
        "Data parallel degree",
        "Tensor model parallel degree",
        "Pipeline parallel degree",
        "Microbatches",
        "Simulated time",
        "PyTorch backend time",
    ]

    with open("mlp_benchmark_dgx_simulation.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        # for (d, t, p, k, dim, layers) in configs:
        for (d, t, p, k, dim, layers) in tqdm.tqdm(configs):
            try:
                assert d > 1 or t > 1 or p > 1
                simulated_time, pytorch_backend_time = benchmark(
                    batch_size,
                    dim,
                    dim,
                    dim,
                    layers,
                    device_throughput,
                    dram_bandwidth,
                    kernel_launch_overhead,
                    network_bandwidth,
                    d,
                    t,
                    p,
                    k,
                )
            except Exception as e:
                traceback.print_exc()
                simulated_time = -1
                pytorch_backend_time = -1
                pure_pytorch_time = -1
            writer.writerow(
                [
                    dim,
                    layers,
                    d,
                    t,
                    p,
                    k,
                    simulated_time,
                    pytorch_backend_time,
                ]
            )
            f.flush()
            torch.cuda.empty_cache()


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
    if args.simulation_parameters_file is not None and os.path.exists(
        args.simulation_parameters_file
    ):
        with open(args.simulation_parameters_file, "rb") as f:
            simulation_parameters = pickle.load(f)
        print(
            f"Reading simulation parameters from {args.simulation_parameters_file}..."
        )
        args.device_throughput = simulation_parameters["device_throughput"]
        args.dram_bandwidth = simulation_parameters["dram_bandwidth"]
        args.kernel_launch_overhead = simulation_parameters["kernel_launch_overhead"]
        args.network_bandwidth = simulation_parameters["network_bandwidth"]
    else:
        simulation_parameters = {}
    update_simulation_parameters = False
    if args.calibrate_device_parameters and (
        args.mode == "simulate"
        or args.mode == "grid_search"
        or args.mode == "distributed_grid_search"
    ):
        print("Calibrating device parameters...")
        (
            args.dram_bandwidth,
            args.device_throughput,
            args.kernel_launch_overhead,
        ) = calibrate_device_parameters()
        simulation_parameters["dram_bandwidth"] = args.dram_bandwidth
        simulation_parameters["device_throughput"] = args.device_throughput
        simulation_parameters["kernel_launch_overhead"] = args.kernel_launch_overhead
        update_simulation_parameters = True
        print(f"DRAM bandwidth: {args.dram_bandwidth:.2e}")
        print(f"Device throughput: {args.device_throughput:.2e}")
        print(f"Kernel launch overhead: {args.kernel_launch_overhead:.2e}")
    if args.calibrate_network_bandwidth and (
        args.mode == "simulate"
        or args.mode == "grid_search"
        or args.mode == "distributed_grid_search"
    ):
        args.network_bandwidth = calibrate_network_bandwidth()
        simulation_parameters["network_bandwidth"] = args.network_bandwidth
        print(f"Network bandwidth: {args.network_bandwidth}")
        update_simulation_parameters = True
    if update_simulation_parameters and args.simulation_parameters_file is not None:
        with open(args.simulation_parameters_file, "wb") as f:
            pickle.dump(simulation_parameters, f)
    if args.mode == "grid_search":
        grid_search(
            args.device_throughput,
            args.dram_bandwidth,
            args.kernel_launch_overhead,
        )
    elif args.mode == "distributed_grid_search":
        distributed_grid_search(
            args.device_throughput,
            args.dram_bandwidth,
            args.kernel_launch_overhead,
            args.network_bandwidth,
        )
    elif args.mode == "simulate":
        x, z, weights = get_inputs(
            args.batch_size, args.dim, args.dim, args.dim, args.layers
        )
        simulated_time, peak_memory = mlp_dist_ir_simulation(
            args.batch_size,
            Gargs.dim,
            args.dim,
            args.dim,
            args.layers,
            x,
            z,
            weights,
            args.device_throughput,
            args.dram_bandwidth,
            args.kernel_launch_overhead,
            args.network_bandwidth,
            args.d,
            args.t,
            args.p,
            args.k,
            verbose=args.verbose,
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
            args.d,
            args.t,
            args.p,
            args.k,
            warmup_steps=args.warmup_steps,
            active_steps=args.active_steps,
            profile=args.profile,
            verbose=args.verbose,
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
        choices=[
            "grid_search",
            "distributed_grid_search",
            "pytorch",
            "simulate",
            "backend",
        ],
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--dim", type=int, default=256, help="Weight dim")
    parser.add_argument("--layers", type=int, default=16, help="# layers")
    parser.add_argument("--warmup_steps", type=int, default=5, help="# warmup steps")
    parser.add_argument("--active_steps", type=int, default=100, help="# active steps")
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
        "--simulation_parameters_file",
        type=str,
        default=None,
        help="File to load/save simulation parameters from/to",
    )
    parser.add_argument("--profile", action="store_true", default=False, help="Profile")
    parser.add_argument(
        "--device_throughput", type=float, default=1.4e13, help="Device throughput"
    )
    parser.add_argument(
        "--dram_bandwidth", type=float, default=9e11, help="DRAM Bandwidth"
    )
    parser.add_argument(
        "--network_bandwidth", type=float, default=64, help="Network bandwidth in Gbps"
    )
    parser.add_argument(
        "--kernel_launch_overhead",
        type=float,
        default=1e-5,
        help="Kernel launch overhead",
    )
    parser.add_argument("-d", type=int, default=1, help="Data parallel degree")
    parser.add_argument("-t", type=int, default=1, help="Tensor model parallel degree")
    parser.add_argument("-p", type=int, default=1, help="Pipeline parallel degree")
    parser.add_argument("-k", type=int, default=1, help="# microbatches")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    args = parser.parse_args()
    main(args)
