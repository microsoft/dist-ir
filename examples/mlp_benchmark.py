import csv
import itertools
import numpy as np
import time
import torch
import tqdm

from dist_ir.ir import cpprint
from dist_ir.backend.torch import run_pytorch
from dist_ir.executor import CostModel, Simulator, SequentialExecutor, infer_types
from dist_ir.transforms import mlp_dhp_transform
from examples import mlp

torch.manual_seed(42)


def get_inputs(batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers):
    x = np.random.normal(size=(batch_size, input_dim))
    z = np.random.normal(size=(batch_size, output_dim))
    weights = [np.random.normal(size=(input_dim, hidden_dim))]
    for i in range(num_hidden_layers - 2):
        weights.append(np.random.normal(size=(hidden_dim, hidden_dim)))
    weights.append(np.random.normal(size=(hidden_dim, output_dim)))
    return x, z, weights


def mlp_dist_ir(
    batch_size,
    input_dim,
    hidden_dim,
    output_dim,
    num_hidden_layers,
    x,
    z,
    weights,
    max_memory_gb=10,
    active_steps=100,
    warmup_steps=5,
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
    init_fn, fn = mlp_dhp_transform(
        fn,
        1,
        1,
        1,
        1,
        topology.devices,
    )
    init_fn = infer_types(init_fn, init_fn.inputs)
    fn = infer_types(fn, init_fn.outputs)
    assert len(fn.inputs) == len(weights) + 2
    input_types = tuple(inp.type for inp in fn.inputs)
    simulator = Simulator(CostModel(topology))
    simulation = simulator.interpret(fn, input_types)
    simulated_time = max([simulation.timestamps[d] for d in simulation.timestamps])
    peak_memory = max([simulation.peak_memory[d] for d in simulation.peak_memory])
    if peak_memory / (1024 ** 3) > max_memory_gb:
        return -1, -1
    seq_executor = SequentialExecutor("numpy")
    input_data = [x, z] + weights
    dist_input_data = seq_executor.compute(init_fn, input_data)
    dist_input_data = tuple(torch.tensor(t) for t in dist_input_data)
    # assert all(t.shape == v.type.shape for (t, v) in zip(dist_input_data, fn.inputs))

    # Measure actual execution time
    per_rank_outputs, runtimes = run_pytorch(
        fn,
        dist_input_data,
        use_gpu=True,
        num_repetitions=active_steps,
        num_warmup=warmup_steps,
    )
    # TODO or median of max?
    actual_time = max(np.median(times) for times in runtimes)

    gradients = [
        per_rank_outputs[0][i] for i, v in enumerate(fn.outputs) if "dw" in v.name
    ]

    return gradients, simulated_time, actual_time


def mlp_pytorch(x, z, weights, warmup_steps=5, active_steps=100):
    batch_size = x.shape[0]
    x = torch.from_numpy(x).cuda()
    z = torch.from_numpy(z).cuda()
    weights = [torch.from_numpy(w).cuda() for w in weights]
    times = []

    for i in range(warmup_steps + active_steps):
        x_ = x.clone()
        z_ = z.clone()
        activations = [x_]
        matmul_outputs = []
        torch.cuda.empty_cache()
        start = time.time()
        for w_ in weights:
            x_ = torch.matmul(x_, w_)
            matmul_outputs.append(x_)
            x_[x_ < 0] = 0
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
        torch.cuda.synchronize()
        times.append(time.time() - start)
    return gradients, np.median(times[warmup_steps:])


def benchmark(
    batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, check_output=True
):
    x, z, weights = get_inputs(
        batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers
    )
    dist_ir_gradients, simulated_time, pytorch_backend_time = mlp_dist_ir(
        batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, x, z, weights
    )
    if simulated_time == -1 or pytorch_backend_time == -1:
        return -1, -1, -1
    torch.cuda.empty_cache()
    pytorch_gradients, pure_pytorch_time = mlp_pytorch(x, z, weights)

    for x, y in zip(pytorch_gradients, dist_ir_gradients):
        np.testing.assert_array_almost_equal(
            x.detach().cpu().numpy(), y.detach().cpu().numpy(), decimal=2
        )

    return simulated_time, pytorch_backend_time, pure_pytorch_time


def main():
    all_batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    all_dims = [16, 32, 64, 128, 256, 512, 1024, 2048]
    all_num_hidden_layers = [4, 8, 16]
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
                    batch_size, dim, dim, dim, num_hidden_layers
                )
            except Exception as e:
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


if __name__ == "__main__":
    main()
