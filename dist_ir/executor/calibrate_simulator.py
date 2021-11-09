# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
from sklearn.linear_model import LinearRegression
import logging
import numpy as np
import time
import torch
from tqdm import tqdm
import pandas as pd

from ..ir import FunctionMaker, Topology, cpprint
from ..ir.type import Device, Float16, Float32, Tensor
from ..backend.torch import run_pytorch
from .type_inference import infer_types
from .cost_model import CostModel
from .simulator import Simulator

BYTES_IN_Gb = 1.25e8
NUM_WARMUP = 5
NUM_REPETITIONS = 30


def _matmul(batch_size, input_dim, output_dim, device, dtype):
    fn = FunctionMaker(name="matmul")
    x = fn.add_input_value(
        "x",
        Tensor(shape=(batch_size, input_dim), dtype=dtype(), device=device),
    )
    w = fn.add_input_value(
        "w",
        Tensor(shape=(input_dim, output_dim), dtype=dtype(), device=device),
    )
    y = fn.add_op(op_type="MatMul", inputs=[x, w], output_names=["y"])
    return fn.finalize()


def _send(src, dst, dtype, m=1024, n=1024):
    fn = FunctionMaker(name=f"send_{src.device_id}_to_{dst.device_id}")
    x = fn.add_input_value("x", Tensor(shape=(m, n), dtype=dtype(), device=src))
    y = fn.add_op(
        op_type="Send", inputs=[x], attributes={"device": dst}, output_names=["y"]
    )
    return fn.finalize()


def _allreduce(devices, dtype, m=1024, n=1024):
    fn = FunctionMaker(name=f"allreduce")
    xs = [
        fn.add_input_value(
            f"x{i}", Tensor(shape=(m, n), dtype=dtype(), device=devices[i])
        )
        for i in range(len(devices))
    ]
    ys = fn.add_op(
        op_type="MPIAllreduce",
        inputs=xs,
        output_names=[f"y{i}" for i in range(len(xs))],
    )
    return fn.finalize()


def _memcpy(rank):
    # TODO: Use this to measure CPU -> GPU transfer time
    t = torch.randn(size=(8192, 8192), dtype=torch.float32)
    start = time.time()
    t = t.to(f"cuda:{rank}")
    torch.cuda.synchronize()
    latency = time.time() - start
    size_in_bytes = t.element_size() * t.nelement()
    return size_in_bytes / BYTES_IN_Gb / latency


def calibrate_network_bandwidth(dtype):
    """Estimates network bandwidth by sending tensors between every pair of GPUs."""
    dist_ir_dtype = Float32 if dtype == "fp32" else Float16
    pytorch_dtype = torch.float32 if dtype == "fp32" else torch.float16
    bandwidths = []
    size = 16384
    max_input = torch.randn(size=(size, size), dtype=pytorch_dtype)
    params = {}
    devices = [Device(i + 1, "gpu") for i in range(torch.cuda.device_count())]
    for i, src in enumerate(devices):
        # TODO: Calibrate CPU to GPU transfer time properly
        bandwidths.append([0, src.device_id, 64])
        for dst in devices[i + 1 :]:
            fn = _send(src, dst, dist_ir_dtype, m=size, n=size)
            fn = infer_types(fn, fn.inputs)
            results = run_pytorch(
                fn=fn,
                inputs=[max_input[:size, :size]],
                use_gpu=True,
                num_repetitions=NUM_REPETITIONS,
                num_warmup=NUM_WARMUP,
            )
            pytorch_latency = results.latency
            torch.cuda.empty_cache()
            bandwidth = fn.inputs[0].type.size() / BYTES_IN_Gb / pytorch_latency
            bandwidths.append([src.device_id, dst.device_id, bandwidth])
            bandwidths.append([dst.device_id, src.device_id, bandwidth])
            logging.info(
                f"bandwidth[({src.device_id}, {dst.device_id})] = {bandwidth} Gbps"
            )

    return bandwidths


def calibrate_device_parameters(dtype):
    """Estimates DRAM bandwidth, device throughput, and kernel launch overhead using MatMul ops."""
    dist_ir_dtype = Float32 if dtype == "fp32" else Float16
    pytorch_dtype = torch.float32 if dtype == "fp32" else torch.float16
    all_batch_sizes = [2 ** i for i in range(12, 14)]
    all_input_dims = [2 ** i for i in range(12, 14)]
    all_output_dims = [2 ** i for i in range(12, 14)]
    if dtype == "fp16":
        all_batch_sizes = [2 * v for v in all_batch_sizes]
        all_input_dims = [2 * v for v in all_input_dims]
        all_output_dims = [2 * v for v in all_output_dims]
    n = len(all_batch_sizes) * len(all_input_dims) * len(all_output_dims)
    X = np.zeros(shape=(n, 3))
    Y = np.zeros(shape=(n,))
    data = []
    device = Device(0, "gpu")
    max_inputs = [
        torch.randn(
            size=(max(all_batch_sizes), max(all_input_dims)), dtype=pytorch_dtype
        ),
        torch.randn(
            size=(max(all_input_dims), max(all_output_dims)), dtype=pytorch_dtype
        ),
    ]
    for i, (batch_size, input_dim, output_dim) in enumerate(
        tqdm(list(itertools.product(all_batch_sizes, all_input_dims, all_output_dims)))
    ):
        fn = _matmul(batch_size, input_dim, output_dim, device, dist_ir_dtype)
        x = fn.inputs[0].type
        y = fn.inputs[1].type
        data_size = x.dtype.size() * (x.shape[0] * x.shape[1] + y.shape[0] * y.shape[1])
        flops = (2 * x.shape[1] - 1) * x.shape[0] * y.shape[1]
        X[i][0] = data_size
        X[i][1] = flops
        X[i][2] = 1

        results = run_pytorch(
            fn=fn,
            inputs=[
                max_inputs[0][:batch_size, :input_dim],
                max_inputs[1][:input_dim, :output_dim],
            ],
            use_gpu=True,
            num_repetitions=NUM_REPETITIONS,
            num_warmup=NUM_WARMUP,
        )
        pytorch_latency = results.latency
        Y[i] = pytorch_latency
        data.append(
            {
                "m": batch_size,
                "n": input_dim,
                "k": output_dim,
                "data_size": data_size,
                "flops": flops,
                "latency": pytorch_latency,
            }
        )
        torch.cuda.empty_cache()

    df = pd.DataFrame(data)
    df.to_csv("matmul_benchmark.csv")

    reg = LinearRegression(positive=True, fit_intercept=False).fit(X, Y)

    return (reg.coef_[0], reg.coef_[1], reg.coef_[2])


def calibrate_allreduce_parameters(dtype):
    """Estimates a linear function of 2 parameters (+launch overhead) to model allreduce cost."""
    dist_ir_dtype = Float32 if dtype == "fp32" else Float16
    pytorch_dtype = torch.float32 if dtype == "fp32" else torch.float16
    all_input_dims = [2 ** i for i in range(11, 14)]
    all_output_dims = [2 ** i for i in range(11, 14)]
    max_input = torch.randn(
        size=(max(all_input_dims), max(all_output_dims)), dtype=pytorch_dtype
    )
    n = len(all_input_dims) * len(all_output_dims)
    X = np.zeros(shape=(n, 3))
    Y = np.zeros(shape=(n,))
    params = {}
    devices = [Device(0, "cpu")] + [
        Device(i + 1, "gpu") for i in range(torch.cuda.device_count())
    ]
    all_num_devices = [2 ** i for i in range(1, int(np.log2(len(devices))) + 1)]
    for num_devices in all_num_devices:
        for i, (input_dim, output_dim) in enumerate(
            tqdm(list(itertools.product(all_input_dims, all_output_dims)))
        ):
            fn = _allreduce(
                devices[1 : num_devices + 1], dist_ir_dtype, input_dim, output_dim
            )
            fn = infer_types(fn, fn.inputs)
            X[i][0] = fn.inputs[0].type.size() / BYTES_IN_Gb
            X[i][1] = num_devices
            X[i][2] = 1

            results = run_pytorch(
                fn=fn,
                inputs=[
                    max_input[:input_dim, :output_dim] for _ in range(len(fn.inputs))
                ],
                use_gpu=True,
                num_repetitions=NUM_REPETITIONS,
                num_warmup=NUM_WARMUP,
            )
            pytorch_latency = results.latency
            Y[i] = pytorch_latency
            torch.cuda.empty_cache()

        reg = LinearRegression(positive=True, fit_intercept=False).fit(X, Y)
        params[num_devices] = (reg.coef_[0], reg.coef_[1], reg.coef_[2])
    return params
