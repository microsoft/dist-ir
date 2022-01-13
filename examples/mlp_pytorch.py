# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import numpy as np
import pandas as pd
import time
import torch
import tqdm

from queue import Queue

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dist_ir.utils import constants
from examples.mlp import run_mlp


class MLPTorch(torch.nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_hidden_layers, dtype, weight=None
    ):
        super().__init__()
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


"""
def train(mlp, x, z, num_warmup, num_repetitions, rank=0):
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
        rtency
        wnnuntimes.append(runtime)
    grads = [w.grad for w in mlp.params]
    return grads, runtimes[num_warmup:]
"""


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # print(f"Initialized rank {rank}")


def cleanup():
    dist.destroy_process_group()


def driver(
    rank,
    world_size,
    batch_size,
    input_dim,
    hidden_dim,
    output_dim,
    num_hidden_layers,
    dtype,
    num_warmup,
    num_repetitions,
    result_queue,
):
    torch.cuda.set_device(rank)

    if world_size > 1:
        setup(rank, world_size)

    if dtype == "fp16":
        np_dtype = np.float16
    elif dtype == "fp32":
        np_dtype = np.float

    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.02, size=(batch_size // world_size, input_dim)).astype(np_dtype)
    z = rng.normal(0, 0.02, size=(batch_size // world_size, output_dim)).astype(
        np_dtype
    )
    weight = rng.normal(0, 0.02, size=(hidden_dim, hidden_dim)).astype(np_dtype)

    x = torch.from_numpy(x).cuda()
    z = torch.from_numpy(z).cuda()

    mlp = MLPTorch(
        input_dim,
        hidden_dim,
        output_dim,
        num_hidden_layers,
        dtype,
        torch.from_numpy(weight),
    ).cuda()

    if world_size > 1:
        mlp = DDP(mlp, device_ids=[rank])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=mlp.parameters(), lr=0)
    runtimes = []
    torch.cuda.synchronize()
    if world_size > 1:
        torch.distributed.barrier()
    for i in list(range(num_warmup + num_repetitions)):
        # print(f"Starting iteration {i+1} / {num_warmup + num_repetitions}...")
        start = time.time()
        optimizer.zero_grad()
        z_pred = mlp(x)
        loss = criterion(z, z_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if world_size > 1:
            torch.distributed.barrier()
        runtime = time.time() - start
        runtimes.append(runtime)

    # grads = [w.grad for w in mlp.params]
    # return grads, runtimes[num_warmup:]

    # TODO: Return weights and/or gradients?
    result_queue.put((runtimes, None))

    if world_size > 1:
        cleanup()


def experiment(
    batch_size,
    input_dim,
    hidden_dim,
    output_dim,
    num_hidden_layers,
    dtype,
    num_warmup,
    num_repetitions,
    world_size=1,
):
    if world_size > 1:
        smp = mp.get_context("spawn")
        result_queue = smp.SimpleQueue()
        mp.spawn(
            driver,
            args=(
                world_size,
                batch_size,
                input_dim,
                hidden_dim,
                output_dim,
                num_hidden_layers,
                dtype,
                num_warmup,
                num_repetitions,
                result_queue,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        result_queue = Queue()
        driver(
            0,
            world_size,
            batch_size,
            input_dim,
            hidden_dim,
            output_dim,
            num_hidden_layers,
            dtype,
            num_warmup,
            num_repetitions,
            result_queue,
        )
    pytorch_runtimes = np.zeros((world_size, num_warmup + num_repetitions))
    for rank in range(world_size):
        (runtimes, _) = result_queue.get()
        pytorch_runtimes[rank] = runtimes

    # print(f"PyTorch runtimes:")
    # print(pytorch_runtimes)

    """
    pytorch_outputs, pytorch_runtimes = train(
        mlp,
        torch.from_numpy(x).cuda(),
        torch.from_numpy(z).cuda(),
        num_warmup,
        num_repetitions,
    )
    """

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
        dp_degree=world_size,
        hp_degree=1,
        pp_degree=1,
        num_microbatches=1,
        device_throughput=constants.DEFAULT_DEVICE_THROUGHPUT,
        dram_bandwidth=constants.DEFAULT_DRAM_BANDWIDTH,
        kernel_launch_overhead=constants.DEFAULT_KERNEL_LAUNCH_OVERHEAD,
        network_bandwidth=constants.DEFAULT_NETWORK_BANDWIDTH,
        num_warmup=num_warmup,
        num_repetitions=num_repetitions,
        skip_allgathers=True,
    )
    pytorch_latencies = [
        np.max(pytorch_runtimes[:, i]) for i in range(len(pytorch_runtimes[0]))
    ]
    pytorch_latency = np.median(pytorch_latencies)
    dist_ir_latency = dist_ir_results.latency

    print(f"World size: {world_size}")
    print(f"Batch size: {batch_size}")
    print(f"PyTorch throughput: {batch_size / pytorch_latency}")
    print(f"DistIR throughput: {batch_size / dist_ir_results.latency}")
    print()

    # TODO: Verify outputs match
    # dist_ir_output = dist_ir_results.per_rank_outputs[0][0]

    # print(pytorch_output)
    # print(dist_ir_output)
    # print(np.linalg.norm((pytorch_output.cpu() - dist_ir_output).detach().numpy()))

    return dist_ir_latency, pytorch_latency


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(1.0)
    parser = argparse.ArgumentParser(description="MLP PyTorch benchmark")
    parser.add_argument("--dim", type=int, default=4096, help="Weight dimension")
    parser.add_argument(
        "--num_hidden_layers", type=int, default=16, help="Number of hidden layers"
    )
    parser.add_argument(
        "--dtype", choices=["fp16", "fp32"], default="fp16", help="Data type"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Run n-way data parallel",
    )
    args = parser.parse_args()
    world_sizes = [1]
    if args.distributed:
        world_sizes.append(torch.cuda.device_count())
    data = []
    for world_size in world_sizes:
        if world_size == 1:
            batch_sizes = [2 ** i for i in range(7, 13)]
        else:
            batch_sizes = [2 ** i for i in range(7, 15)]
        for batch_size in batch_sizes:
            dist_ir_latency, pytorch_latency = experiment(
                batch_size,
                args.dim,
                args.dim,
                args.dim,
                args.num_hidden_layers,
                args.dtype,
                5,
                10,
                world_size,
            )
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            data.append((world_size, batch_size, dist_ir_latency, pytorch_latency))
    df = pd.DataFrame(
        data, columns=["world_size", "batch_size", "dist_ir_latency", "pytorch_latency"]
    )
    df.to_csv("pytorch_backend_benchmark.csv")
