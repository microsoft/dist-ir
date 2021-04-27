import os
from tempfile import TemporaryDirectory
from typing import Any, Tuple

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


def run_multiprocesses(
    per_rank_functions: Tuple[Function], per_rank_inputs: Tuple[Any], backend="gloo"
):
    assert len(per_rank_functions) == len(per_rank_inputs)
    world_size = len(per_rank_functions)

    # Convert per-rank DistIR functions to torch.nn.Modules:
    per_rank_modules = list(map(function_to_module, per_rank_functions))
    for d, gm in enumerate(per_rank_modules):
        print(f"{d}\n{gm.graph}\n")

    io_dir = TemporaryDirectory()
    # print("run_multiprocess: saving I/O to:", io_dir.name)

    def run_process(rank, module):
        """Initialize the distributed environment."""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend, rank=rank, world_size=world_size)

        per_rank_inputs = torch.load(os.path.join(io_dir.name, f"in.{rank}.pt"))

        # TODO time the next line only
        res = module(*per_rank_inputs)

        torch.save(res, os.path.join(io_dir.name, f"out.{rank}.pt"))

    # Save inputs for each per-rank function:
    # TODO lowered pytorch file numbers devices 0...num_devices-1
    for d, inps in enumerate(per_rank_inputs):
        torch.save(inps, os.path.join(io_dir.name, f"in.{d}.pt"))

    processes = []
    for rank, per_rank_module in enumerate(per_rank_modules):
        p = Process(target=run_process, args=(rank, per_rank_module))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Load outputs:
    per_rank_outputs = [
        torch.load(os.path.join(io_dir.name, f"out.{d}.pt")) for d in range(world_size)
    ]
    io_dir.cleanup()

    return per_rank_outputs
