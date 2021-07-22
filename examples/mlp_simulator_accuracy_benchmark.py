import numpy as np
import time
import torch

from dist_ir.backend.torch import run_pytorch
from dist_ir.executor import SequentialExecutor, infer_types
from dist_ir.transforms import mlp_dhp_transform
from examples import mlp

torch.manual_seed(42)


def get_inputs(batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers):
    x = np.random.normal(size=(batch_size, input_dim))
    z = np.random.normal(size=(batch_size, output_dim))
    weights = [np.random.normal(size=(input_dim, hidden_dim))]
    for i in range(num_hidden_layers - 1):
        weights.append(np.random.normal(size=(hidden_dim, hidden_dim)))
    weights.append(np.random.normal(size=(hidden_dim, output_dim)))
    return x, z, weights


def mlp_pytorch(x, z, weights, warmup_steps=5, active_steps=10):
    batch_size = x.shape[0]
    x = torch.from_numpy(x).cuda()
    z = torch.from_numpy(z).cuda()
    weights = [torch.from_numpy(w).cuda() for w in weights]
    times = []

    for i in range(warmup_steps + active_steps):
        x_ = x
        z_ = z
        start = time.time()
        activations = [x_]
        for w_ in weights:
            x_ = torch.matmul(x_, w_)
            x_[x_ < 0] = 0
            activations.append(x_)

        loss = torch.square(x_ - z_) / batch_size
        dx_ = 2 * (x_ - z_) / batch_size

        gradients = []
        for j, w_ in enumerate(reversed(weights)):
            x_ = activations[len(activations) - j - 1]
            dx_, dw_ = torch.matmul(dx_, w_.T), torch.matmul(x_.T, dx_)
            gradients.append(dw_)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    return np.median(times[warmup_steps:])


def mlp_dist_ir(
    batch_size,
    input_dim,
    hidden_dim,
    output_dim,
    num_hidden_layers,
    x,
    z,
    weights,
    warmup_steps=5,
    active_steps=10,
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
    seq_executor = SequentialExecutor("numpy")
    input_data = [x, z] + weights
    dist_input_data = seq_executor.compute(init_fn, input_data)
    dist_input_data = tuple(torch.tensor(t) for t in dist_input_data)
    # assert all(t.shape == v.type.shape for (t, v) in zip(dist_input_data, fn.inputs))

    # Measure actual execution time
    # TODO check outputs match?
    _, runtimes = run_pytorch(
        fn,
        dist_input_data,
        use_gpu=True,
        num_repetitions=active_steps,
        num_warmup=warmup_steps,
    )
    # TODO or median of max?
    actual_time = max(np.median(times) for times in runtimes)
    return actual_time


def main():
    x, z, weights = get_inputs(128, 128, 128, 128, 4)
    pytorch_time = mlp_pytorch(x, z, weights)
    dist_ir_time = mlp_dist_ir(128, 128, 128, 128, 4, x, z, weights)
    print(f"PyTorch time: {pytorch_time * 1e3:.2f} ms")
    print(f"DistIR time: {dist_ir_time * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
