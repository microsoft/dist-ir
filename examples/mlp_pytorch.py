import numpy as np
import pandas as pd
import time
import torch
import tqdm

from dist_ir.utils import constants
from examples.mlp import run_mlp


class MLPTorch(torch.nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_hidden_layers, dtype, weight=None
    ):
        super().__init__()
        """
        self.register_parameter(
            name="w0",
            param=torch.nn.Parameter(weight),
        )
        for i in range(1, num_hidden_layers - 1):
            self.register_parameter(
                name=f"w{i}",
                param=torch.nn.Parameter(weight),
            )
        self.register_parameter(
            name=f"w{i+1}",
            param=torch.nn.Parameter(weight),
        )
        """
        if weight is not None:
            params = []
            params.append(torch.nn.Parameter(weight))
            for i in range(1, num_hidden_layers - 1):
                params.append(torch.nn.Parameter(weight))
            params.append(torch.nn.Parameter(weight))
            self.params = torch.nn.ParameterList(params)
        else:
            raise NotImplementedError("Only pre-initialized weight accepted")

    def forward(self, x):
        for i, w in enumerate(self.params):
            x = torch.relu(torch.matmul(x, w))
        return x


def train(mlp, x, z, num_warmup, num_repetitions):
    # def train(mlp, x, num_warmup, num_repetitions):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=mlp.parameters(), lr=0)
    runtimes = []
    torch.cuda.synchronize()
    for i in list(range(num_warmup + num_repetitions)):
        start = time.time()
        optimizer.zero_grad()
        z_pred = mlp(x)
        loss = criterion(z, z_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        runtime = time.time() - start
        runtimes.append(runtime)
    grads = [w.grad for w in mlp.params]
    return grads, runtimes[num_warmup:]


def experiment(
    batch_size,
    input_dim,
    hidden_dim,
    output_dim,
    num_hidden_layers,
    dtype,
    num_warmup,
    num_repetitions,
):
    if dtype == "fp16":
        np_dtype = np.float16
    elif dtype == "fp32":
        np_dtype = np.float

    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.02, size=(batch_size, input_dim)).astype(np_dtype)
    z = rng.normal(0, 0.02, size=(batch_size, output_dim)).astype(np_dtype)
    weight = rng.normal(0, 0.02, size=(hidden_dim, hidden_dim)).astype(np_dtype)

    mlp = MLPTorch(
        input_dim,
        hidden_dim,
        output_dim,
        num_hidden_layers,
        dtype,
        torch.from_numpy(weight),
    ).cuda()
    pytorch_outputs, pytorch_runtimes = train(
        mlp,
        torch.from_numpy(x).cuda(),
        torch.from_numpy(z).cuda(),
        num_warmup,
        num_repetitions,
    )

    dist_ir_results = run_mlp(
        phase="training",
        backend="pytorch",
        dtype=dtype,
        use_gpu=True,
        batch_size=batch_size,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_hidden_layers=num_hidden_layers,
        dp_degree=1,
        hp_degree=1,
        pp_degree=1,
        num_microbatches=1,
        device_throughput=constants.DEFAULT_DEVICE_THROUGHPUT,
        dram_bandwidth=constants.DEFAULT_DRAM_BANDWIDTH,
        kernel_launch_overhead=constants.DEFAULT_KERNEL_LAUNCH_OVERHEAD,
        network_bandwidth=constants.DEFAULT_NETWORK_BANDWIDTH,
        num_warmup=num_warmup,
        num_repetitions=num_repetitions,
    )

    pytorch_latency = np.median(pytorch_runtimes)
    dist_ir_latency = dist_ir_results.latency

    print(f"PyTorch latency: {pytorch_latency * 1e3} ms")
    print(f"DistIR latency: {dist_ir_results.latency * 1e3} ms")

    return dist_ir_latency, pytorch_latency

    # dist_ir_output = dist_ir_results.per_rank_outputs[0][0]

    # print(pytorch_output)
    # print(dist_ir_output)
    # print(np.linalg.norm((pytorch_output.cpu() - dist_ir_output).detach().numpy()))


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(1.0)
    data = []
    for batch_size in [2 ** i for i in range(7, 18)]:
        try:
            dist_ir_latency, pytorch_latency = experiment(
                batch_size, 8192, 8192, 8192, 16, "fp16", 5, 10
            )
        except RuntimeError as e:
            break
        data.append((batch_size, dist_ir_latency, pytorch_latency))
    df = pd.DataFrame(data, columns=["batch_size", "dist_ir_latency", "pytorch_latency"])
    df.to_csv("pytorch_backend_benchmark.csv")
