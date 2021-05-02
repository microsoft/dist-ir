from functools import partial
from operator import getitem
import os
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import fx

from ..executor.rank_projector import project
from ..ir import Function


# TODO kwargs of these functions are required, enforce this somewhere
def _allgather(x_i, dim=0):
    world_size = dist.get_world_size()
    xs = [torch.zeros_like(x_i) for _ in range(world_size)]
    dist.all_gather(xs, x_i)
    x = torch.cat(xs, dim=dim)
    return x


def _allreduce(x):
    dist.all_reduce(x)
    return x


def _concat2(x, y, dim=None):
    return torch.cat((x, y), dim=dim)


def _identity(x):
    return x


def _loss(x, y, N=None):
    return torch.square(x - y) / N


def _loss_grad(x, y, N=None):
    return 2 * (x - y) / N


def _matmul_grad(x, y, dz):
    return (torch.matmul(dz, y.T), torch.matmul(x.T, dz))


def _recv(shape=None, device=None):
    x = torch.zeros(shape)
    # TODO pytorch rank = device_id - 1
    dist.recv(x, device - 1)
    return x


def _relu_grad(x, dy):
    # TODO: fix
    dx = torch.zeros(dy.shape)
    dx[dy > 0] = 1
    return dx


def _send(x, device=None):
    # TODO pytorch rank = device_id - 1
    dist.send(x, device - 1)


_op_to_torch = {
    "Add": torch.add,
    "Concat": _concat2,
    "Identity": _identity,
    "Loss": _loss,
    "LossGrad": _loss_grad,
    "MatMul": torch.matmul,
    "MatMulGrad": _matmul_grad,
    "RecvP2P": _recv,
    "Relu": torch.relu,
    "ReluGrad": _relu_grad,
    "SendP2P": _send,
    "MPIAllgather": _allgather,
    "MPIAllreduce": _allreduce,
}


def function_to_module(fn: Function) -> torch.nn.Module:
    g = fx.Graph()
    value_map = {}

    # TODO need to check that fn has unique value names

    # Convert inputs
    for v in fn.inputs:
        value_map[v] = g.placeholder(v.name)

    # Convert ops
    for op in fn.ops:
        inputs = tuple(value_map[v] for v in op.inputs)
        kwargs = None if op.attributes is None else {**op.attributes}
        output = g.call_function(_op_to_torch[op.op_type], inputs, kwargs)
        if len(op.outputs) > 1:
            for i, v in enumerate(op.outputs):
                value_map[v] = g.call_function(getitem, (output, i))
        elif len(op.outputs) == 1:
            value_map[op.outputs[0]] = output

    # Convert outputs
    g.output(tuple(value_map[v] for v in fn.outputs))

    return fx.GraphModule({}, g)


def run_process(
    use_gpu, world_size, io_dir, num_warmup_steps, num_repetitions, rank, fn
):
    """The Python function on rank `rank` that runs module `module`."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    backend = "nccl" if use_gpu else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    per_rank_inputs = torch.load(os.path.join(io_dir.name, f"in.{rank}.pt"))

    # Convert per-rank DistIR function to torch.nn.Module:
    module = function_to_module(fn)

    if use_gpu:
        # Move module and inputs to GPU
        module.to(rank)
        for t in per_rank_inputs:
            t.to(rank)

    events = []

    def add_event():
        if use_gpu:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        else:
            events.append(perf_counter())

    # Time a bunch of executions, then execute once for output values
    add_event()
    for _ in range(num_warmup_steps + num_repetitions):
        res = module(*per_rank_inputs)
        if world_size > 1:
            torch.distributed.barrier()
        add_event()

    torch.save(res, os.path.join(io_dir.name, f"out.{rank}.pt"))

    if use_gpu:
        runtimes = [
            events[i].elapsed_time(events[i + 1]) / 1e3 for i in range(len(events) - 1)
        ]
        torch.cuda.synchronize()
    else:
        runtimes = [events[i + 1] - events[i] for i in range(len(events) - 1)]

    dist.destroy_process_group()
    return runtimes[num_warmup_steps:]


def run_multiprocesses(
    per_rank_functions: Tuple[Function],
    per_rank_inputs: Tuple[Any],
    use_gpu=False,
    num_repetitions=1,
    num_warmup=0,
):
    assert len(per_rank_functions) == len(per_rank_inputs)
    world_size = len(per_rank_functions)

    # Save inputs for each per-rank function:
    io_dir = TemporaryDirectory()
    # print("run_multiprocess: saving I/O to:", io_dir.name)
    # TODO lowered pytorch file numbers devices 0...num_devices-1
    for d, inps in enumerate(per_rank_inputs):
        torch.save(inps, os.path.join(io_dir.name, f"in.{d}.pt"))

    global run_process
    per_rank_runner = partial(
        run_process, use_gpu, world_size, io_dir, num_warmup, num_repetitions
    )
    with torch.multiprocessing.Pool(world_size) as p:
        runtimes = p.starmap(per_rank_runner, enumerate(per_rank_functions))

    # Load outputs:
    per_rank_outputs = [
        torch.load(os.path.join(io_dir.name, f"out.{d}.pt")) for d in range(world_size)
    ]
    io_dir.cleanup()

    return per_rank_outputs, runtimes


def run_pytorch(num_devices, fn, inputs):
    """Project `fn` and run on `inputs` over `num_devices` devices using the
    PyTorch backend.
    """
    # TODO check that fn uses devices [0...num_devices)
    per_rank_fns = project(fn, tuple(v.type for v in fn.inputs), num_devices)
    per_rank_inputs = [[] for _ in range(num_devices)]
    for v, a in zip(fn.inputs, inputs):
        per_rank_inputs[v.type.device.device_id - 1].append(a)
    return run_multiprocesses(per_rank_fns, per_rank_inputs)
