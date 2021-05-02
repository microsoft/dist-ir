from functools import partial
import os
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import fx
from torch.multiprocessing import Process

from ..ir import Function


# TODO at op creation time, enforce MPIAllgather ops attributes
def _allgather(x_i, world_size=None, dim=0):
    xs = [torch.zeros_like(x_i) for _ in range(world_size)]
    dist.all_gather(xs, x_i)
    x = torch.cat(xs, dim=dim)
    return x


_op_to_torch = {
    "MatMul": torch.matmul,
    "Relu": torch.relu,
    "MPIAllgather": _allgather,
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
        assert len(op.outputs) == 1, "TODO how to handle multiple outputs in fx"
        kwargs = None if op.attributes is None else {**op.attributes}
        output = g.call_function(_op_to_torch[op.op_type], inputs, kwargs)
        value_map[op.outputs[0]] = output

    # Convert outputs
    for v in fn.outputs:
        g.output(value_map[v])

    return fx.GraphModule({}, g)


def run_process(
    use_gpu, world_size, io_dir, num_warmup_steps, num_repetitions, rank, module
):
    """The Python function on rank `rank` that runs module `module`."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    backend = "nccl" if use_gpu else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    per_rank_inputs = torch.load(os.path.join(io_dir.name, f"in.{rank}.pt"))

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
    else:
        runtimes = [events[i + 1] - events[i] for i in range(len(events) - 1)]

    torch.cuda.synchronize()
    dist.destroy_process_group()
    return runtimes[num_warmup_steps:]


def run_multiprocesses(
    per_rank_functions: Tuple[Function],
    per_rank_inputs: Tuple[Any],
    use_gpu=False,
    num_repetitions=100,
    num_warmup=10,
):
    assert len(per_rank_functions) == len(per_rank_inputs)
    world_size = len(per_rank_functions)

    # Convert per-rank DistIR functions to torch.nn.Modules:
    per_rank_modules = list(map(function_to_module, per_rank_functions))
    for d, gm in enumerate(per_rank_modules):
        print(f"{d}\n{gm.graph}\n")

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
        runtimes = p.starmap(per_rank_runner, enumerate(per_rank_modules))

    # Load outputs:
    per_rank_outputs = [
        torch.load(os.path.join(io_dir.name, f"out.{d}.pt")) for d in range(world_size)
    ]
    io_dir.cleanup()

    return per_rank_outputs, runtimes
