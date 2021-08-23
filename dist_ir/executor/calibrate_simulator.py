import itertools
from sklearn.linear_model import LinearRegression
import numpy as np
import time
import torch
from tqdm import tqdm

from dist_ir.ir import FunctionMaker, Topology, cpprint
from dist_ir.ir.type import Device, Float32, Tensor
from dist_ir.backend.torch import run_pytorch
from .type_inference import infer_types
from .sequential_executor import SequentialExecutor
from .cost_model import CostModel
from .simulator import Simulator

BYTES_IN_Gb = 1.25e8


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


def _send(src, dst, m=1024, n=1024):
    fn = FunctionMaker(name=f"send_{src.device_id}_to_{dst.device_id}")
    x = fn.add_input_value("x", Tensor(shape=(m, n), dtype=Float32(), device=src))
    y = fn.add_op(
        op_type="Send", inputs=[x], attributes={"device": dst}, output_names=["y"]
    )
    return fn.finalize()


def _allreduce(devices, m=1024, n=1024):
    fn = FunctionMaker(name=f"allreduce")
    xs = [
        fn.add_input_value(
            f"x{i}", Tensor(shape=(m, n), dtype=Float32(), device=devices[i])
        )
        for i in range(len(devices))
    ]
    """
    xs_contention = [
        fn.add_input_value(
            f"x2", Tensor(shape=(8192, 8192), dtype=Float32(), device=devices[2])
        ),
        fn.add_input_value(
            f"x3", Tensor(shape=(8192, 8192), dtype=Float32(), device=devices[2])
        ),
        fn.add_input_value(
            f"x4", Tensor(shape=(8192, 8192), dtype=Float32(), device=devices[3])
        ),
        fn.add_input_value(
            f"x5", Tensor(shape=(8192, 8192), dtype=Float32(), device=devices[3])
        ),
    ]
    """
    ys = fn.add_op(
        op_type="MPIAllreduce",
        inputs=xs,
        output_names=[f"y{i}" for i in range(len(xs))],
    )
    """
    ys_contention = [
        fn.add_op(op_type="MatMul", inputs=xs_contention[:2], output_names=["y2"]),
        fn.add_op(op_type="MatMul", inputs=xs_contention[2:], output_names=["y3"]),
    ]
    """
    """
    ys_contention = fn.add_op(
        op_type="MPIAllreduce",
        inputs=xs_contention,
        output_names=[f"y{i}" for i in range(2, 4)],
    )
    """
    return fn.finalize()


def _memcpy(rank):
    t = torch.randn(size=(8192, 8192), dtype=torch.float32)
    start = time.time()
    t = t.to(f"cuda:{rank}")
    torch.cuda.synchronize()
    latency = time.time() - start
    size_in_bytes = t.element_size() * t.nelement()
    return size_in_bytes / BYTES_IN_Gb / latency


def network_bandwidth_debug():
    # devices = [Device(i + 1, "gpu") for i in range(4)]
    topology = Topology()
    topology.add_device(0, "cpu")
    for i in range(4):
        topology.add_device(i + 1, "gpu")
    for i in range(4):
        for j in range(i + 1, 4):
            topology.set_bandwidth(topology.devices[i + 1], topology.devices[j + 1], 7)
    sizes = [2048, 4096, 8192, 16384]
    # sizes = [32, 64, 128, 256, 1024, 2048, 4096, 8192, 16384]
    for i in range(len(sizes)):
        for j in range(i, len(sizes)):
            m = sizes[i]
            n = sizes[j]
            fn = _allreduce(topology.devices[1:], m, n)
            fn = infer_types(fn, fn.inputs)
            _, runtimes = run_pytorch(
                fn=fn,
                inputs=[
                    torch.randn(size=fn.inputs[i].type.shape, dtype=torch.float32)
                    for i in range(len(fn.inputs))
                ],
                use_gpu=True,
                num_repetitions=10,
                num_warmup=5,
            )
            real_latency = np.median(runtimes[0])
            ex = Simulator(CostModel(topology))
            state = ex.interpret(fn, tuple(inp.type for inp in fn.inputs))
            simulated_latency = np.max([state.timestamps[d] for d in state.timestamps])
            simulated_bandwidth = (
                fn.inputs[0].type.size() / BYTES_IN_Gb / simulated_latency
            )

            print(
                f"{m}x{n}: shape={fn.inputs[0].type.shape}, "
                f"size={fn.inputs[0].type.size()}, real latency={real_latency}, "
                f"simulated latency={simulated_latency}"
            )


def calibrate_network_bandwidth():
    devices = [Device(0, "cpu")] + [
        Device(i + 1, "gpu") for i in range(torch.cuda.device_count())
    ]
    bandwidths = {}
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for i in range(1, len(devices)):
        bandwidths[(0, i)] = _memcpy(i-1)
        print(f"bandwidth[(0, {i})] = {bandwidths[(0, i)]} Gbps")
        for j in range(i + 1, len(devices)):
            X = np.zeros(shape=(len(sizes), 2))
            X[:, 1] = 1
            Y = np.zeros(shape=(len(sizes),))
            for k, size in enumerate(sizes):
                fn = _send(devices[i], devices[j], m=size, n=size)
                X[k][0] = fn.inputs[0].type.size() / BYTES_IN_Gb
                _, runtimes = run_pytorch(
                    fn=fn,
                    inputs=[
                        torch.randn(size=fn.inputs[0].type.shape, dtype=torch.float32),
                    ],
                    use_gpu=True,
                    num_repetitions=10,
                    num_warmup=5,
                )
                pytorch_latency = np.median(runtimes[0])
                Y[k] = pytorch_latency
            reg = LinearRegression(positive=True, fit_intercept=False).fit(X, Y)
            bandwidth = 1.0 / reg.coef_[0]
            kernel_launch_overhead = reg.coef_[1]
            print(
                f"bandwidth[({i}, {j})] = {bandwidth} Gbps, "
                f"kernel_launch_overhead={kernel_launch_overhead}"
            )
            bandwidths[(i, j)] = bandwidth
    return bandwidths


def calibrate_device_parameters():
    all_batch_sizes = [1024, 2048, 4096]
    all_input_dims = [1024, 2048, 4096]
    all_output_dims = [1024, 2048, 4096]
    n = len(all_batch_sizes) * len(all_input_dims) * len(all_output_dims)
    X = np.zeros(shape=(n, 3))
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
        X[i][2] = 1

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

    reg = LinearRegression(positive=True, fit_intercept=False).fit(X, Y)
    return 1.0 / reg.coef_[0], 1.0 / reg.coef_[1], reg.coef_[2]


def calibrate_simulator():
    device_parameters = calibrate_device_parameters()
    network_bandwidth = calibrate_network_bandwidth()
    return (*device_parameters, network_bandwidth)
