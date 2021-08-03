import nevergrad as ng
import itertools
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from dist_ir.ir import FunctionMaker
from dist_ir.ir.type import Float32, Tensor
from dist_ir.executor import CostModel, Simulator
from dist_ir.backend.torch import run_pytorch
from examples.mlp import get_topology


def _matmul(batch_size, input_dim, output_dim, topology):
    fn = FunctionMaker(name="matmul")
    x = fn.add_input_value(
        "x",
        Tensor(
            shape=(batch_size, input_dim), dtype=Float32(), device=topology.devices[0]
        ),
    )
    w = fn.add_input_value(
        "w",
        Tensor(
            shape=(input_dim, output_dim), dtype=Float32(), device=topology.devices[0]
        ),
    )
    y = fn.add_op(op_type="MatMul", inputs=[x, w], output_names=["y"])
    return fn.finalize()


def calibrate_simulator():
    all_batch_sizes = [1024, 2048, 4096]
    all_input_dims = [1024, 2048, 4096]
    all_output_dims = [1024, 2048, 4096]
    n = len(all_batch_sizes) * len(all_input_dims) * len(all_output_dims)
    X = np.zeros(shape=(n, 2))
    Y = np.zeros(shape=(n,))
    topology = get_topology(1)
    for i, (batch_size, input_dim, output_dim) in enumerate(
        tqdm(list(itertools.product(all_batch_sizes, all_input_dims, all_output_dims)))
    ):
        fn = matmul(batch_size, input_dim, output_dim, topology)
        x = fn.inputs[0].type
        y = fn.inputs[1].type
        data_size = x.dtype.size() * (x.shape[0] * x.shape[1] + y.shape[0] * y.shape[1])
        flops = 2 * x.shape[0] * x.shape[1] * y.shape[1]
        X[i][0] = data_size
        X[i][1] = flops

        _, runtimes = run_pytorch(
            fn=fn,
            inputs=[
                torch.randn(size=fn.inputs[0].type.shape),
                torch.randn(size=fn.inputs[1].type.shape),
            ],
            use_gpu=True,
            num_repetitions=10,
            num_warmup=5,
        )
        pytorch_latency = np.median(runtimes[0])
        Y[i] = pytorch_latency

    reg = LinearRegression(positive=True).fit(X, Y)
    print(f"Intercept: {reg.intercept_}")
    return 1.0 / reg.coef_


def main():
    dram_bandwidth, device_throughput = calibrate_simulator()
    print(f"Device throughput: {device_throughput:e}")
    print(f"DRAM bandwidth: {dram_bandwidth:.2e}")


if __name__ == "__main__":
    main()
