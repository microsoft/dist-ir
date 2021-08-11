import itertools
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from dist_ir.ir import FunctionMaker
from dist_ir.ir.type import Device, Float32, Tensor
from dist_ir.backend.torch import run_pytorch


def _matmul(batch_size, input_dim, output_dim, device):
    fn = FunctionMaker(name="matmul")
    x = fn.add_input_value(
        "x",
        Tensor(shape=(batch_size, input_dim), dtype=Float32(), device=device),
    )
    w = fn.add_input_value(
        "w",
        Tensor(shape=(input_dim, output_dim), dtype=Float32(), device=device),
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
    device = Device(0, "gpu")
    for i, (batch_size, input_dim, output_dim) in enumerate(
        tqdm(list(itertools.product(all_batch_sizes, all_input_dims, all_output_dims)))
    ):
        fn = _matmul(batch_size, input_dim, output_dim, device)
        x = fn.inputs[0].type
        y = fn.inputs[1].type
        data_size = x.dtype.size() * (x.shape[0] * x.shape[1] + y.shape[0] * y.shape[1])
        flops = (2 * x.shape[1] - 1) * x.shape[0] * y.shape[1]
        X[i][0] = data_size
        X[i][1] = flops

        _, runtimes = run_pytorch(
            fn=fn,
            inputs=[
                torch.randn(size=fn.inputs[0].type.shape, dtype=torch.float32),
                torch.randn(size=fn.inputs[1].type.shape, dtype=torch.float32),
            ],
            use_gpu=True,
            num_repetitions=10,
            num_warmup=5,
        )
        pytorch_latency = np.median(runtimes[0])
        Y[i] = pytorch_latency

    reg = LinearRegression(positive=True).fit(X, Y)
    return 1.0 / reg.coef_[0], 1.0 / reg.coef_[1], reg.intercept_


def main():
    dram_bandwidth, device_throughput, kernel_launch_overhead = calibrate_simulator()
    print(f"Device throughput: {device_throughput:e}")
    print(f"DRAM bandwidth: {dram_bandwidth:.2e}")
    print(f"Kernel launch overhead: {kernel_launch_overhead}")


if __name__ == "__main__":
    main()
