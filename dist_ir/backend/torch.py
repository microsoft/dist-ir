from functools import partial
from itertools import combinations
import logging
from operator import getitem
import os
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Dict, List, NamedTuple, Tuple

import torch
import torch.distributed as dist
from torch import fx

from ..executor.rank_projector import project
from ..ir import Function


DistributedContext = NamedTuple(
    "DistributedContext", use_gpu=bool, groups=Dict[Tuple[int, int], Any]  # Any->Group
)


def _init_distributed_context(use_gpu):
    """Since torch.distributed's NCCL backed doesn't support P2P communication,
    we create a group for each pair of ranks and use broadcasts to emulate P2P
    send/recv. This method initializes the groups.
    """
    groups = {}
    if use_gpu:
        world_size = dist.get_world_size()
        for i, j in combinations(range(world_size), 2):
            groups[i, j] = dist.new_group([i, j])
    return DistributedContext(use_gpu=use_gpu, groups=groups)


def _add(x, y, ctx=None):
    return torch.add(x, y)


# TODO kwargs of these functions are required, enforce this somewhere
def _allgather(x_i, dim=0, ctx=None):
    world_size = dist.get_world_size()
    xs = [torch.zeros_like(x_i) for _ in range(world_size)]
    if ctx.use_gpu:
        xs = [x.cuda(dist.get_rank()) for x in xs]

    dist.all_gather(xs, x_i)
    x = torch.cat(xs, dim=dim)
    return x


def _allreduce(x, ctx=None):
    dist.all_reduce(x)
    return x


def _concat2(x, y, dim=None, ctx=None):
    return torch.cat((x, y), dim=dim)


def _identity(x, ctx=None):
    return x


def _loss(x, y, N=None, ctx=None):
    return torch.square(x - y) / N


def _loss_grad(x, y, N=None, ctx=None):
    return 2 * (x - y) / N


def _matmul(x, y, ctx=None):
    return torch.matmul(x, y)


def _matmul_grad(x, y, dz, ctx=None):
    return (torch.matmul(dz, y.T), torch.matmul(x.T, dz))


def _recv(shape=None, device=None, ctx=None):
    x = torch.zeros(shape)
    # TODO pytorch rank = device_id - 1
    if ctx.use_gpu:
        x = x.cuda(dist.get_rank())
        src_rank = device - 1
        dst_rank = dist.get_rank()
        group = ctx.groups[tuple(sorted((src_rank, dst_rank)))]
        dist.broadcast(x, src_rank, group=group)
    else:
        dist.recv(x, device - 1)
    return x


def _relu(x, ctx=None):
    return torch.relu(x)


def _relu_grad(x, dy, ctx=None):
    dx = dy.clone()
    dx[x <= 0] = 0
    return dx


def _send(x, device=None, ctx=None):
    # TODO pytorch rank = device_id - 1
    if ctx.use_gpu:
        src_rank = dist.get_rank()
        dst_rank = device - 1
        group = ctx.groups[tuple(sorted((src_rank, dst_rank)))]
        dist.broadcast(x, src_rank, group=group)
    else:
        dist.send(x, device - 1)
    # Note: in a proper backend, might want to concatenate multiple tensors into
    # a single buffer and call a single send op


_op_to_torch = {
    "Add": _add,
    "Concat": _concat2,
    "Identity": _identity,
    "Loss": _loss,
    "LossGrad": _loss_grad,
    "MatMul": _matmul,
    "MatMulGrad": _matmul_grad,
    "RecvP2P": _recv,
    "Relu": _relu,
    "ReluGrad": _relu_grad,
    "SendP2P": _send,
    "MPIAllgather": _allgather,
    "MPIAllreduce": _allreduce,
}

# Some mock communication ops that return zero tensors of appropriate shape
# to be used in the sequential runner for debugging

_mock_world_size = None


def _mock_allgather(x_i, dim=0, ctx=None):
    xs = [torch.zeros_like(x_i) for _ in range(_mock_world_size)]
    x = torch.cat(xs, dim=dim)
    return x


def _mock_allreduce(x, ctx=None):
    return x


def _mock_recv(shape=None, device=None, ctx=None):
    x = torch.zeros(shape)
    return x


def _mock_send(x, device=None, ctx=None):
    pass


_mock_comm_ops = {
    "RecvP2P": _mock_recv,
    "SendP2P": _mock_send,
    "MPIAllgather": _mock_allgather,
    "MPIAllreduce": _mock_allreduce,
}

_mock_op_to_torch = {**_op_to_torch, **_mock_comm_ops}


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


def run_function(
    ctx: DistributedContext,
    rank: int,
    fn: Function,
    inputs: List[Any],
    debug_mock=False,
):
    # TODO free values when no longer needed
    op_to_torch = _mock_op_to_torch if debug_mock else _op_to_torch
    value_map = {}

    # Add inputs to value_map
    for v, x in zip(fn.inputs, inputs):
        value_map[v] = x
    assert len(fn.inputs) == len(inputs)

    # Run ops
    for op in fn.ops:
        first_output = (
            op.outputs[0].name
            if op.outputs is not None and len(op.outputs) > 0
            else "None"
        )
        logging.info(f"{rank}: {first_output} {op.op_type}")
        inputs = tuple(value_map[v] for v in op.inputs)
        kwargs = {} if op.attributes is None else {**op.attributes}
        kwargs["ctx"] = ctx
        output = op_to_torch[op.op_type](*inputs, **kwargs)
        if len(op.outputs) > 1:
            assert isinstance(output, tuple)
            for i, v in enumerate(op.outputs):
                value_map[v] = output[i]
        elif len(op.outputs) == 1:
            value_map[op.outputs[0]] = output
        logging.info(f"{rank}: {first_output} {op.op_type}")

    # Return outputs
    return tuple(value_map[v] for v in fn.outputs)


def run_process(
    use_gpu, world_size, io_dir, num_warmup_steps, num_repetitions, rank, fn
):
    """The Python function on rank `rank` that runs module `module`."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    backend = "nccl" if use_gpu else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    ctx = _init_distributed_context(use_gpu)

    per_rank_inputs = torch.load(os.path.join(io_dir.name, f"in.{rank}.pt"))

    # # Convert per-rank DistIR function to torch.nn.Module:
    # module = function_to_module(fn)

    if use_gpu:
        # Move module and inputs to GPU
        # TODO how to move interpreted non-module code to GPU?
        # module = module.cuda(rank)
        per_rank_inputs = [t.cuda(rank) for t in per_rank_inputs]

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
        # res = module(*per_rank_inputs)
        res = run_function(ctx, rank, fn, per_rank_inputs)
        if world_size > 1:
            torch.distributed.barrier()
        add_event()

    if use_gpu:
        # Move outputs back to cpu
        res = [t.cpu() for t in res]

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


def run_mock_multiprocess(
    per_rank_functions: Tuple[Function],
    per_rank_inputs: Tuple[Any],
    num_repetitions=1,
    num_warmup=0,
):
    assert len(per_rank_functions) == len(per_rank_inputs)
    global _mock_world_size
    _mock_world_size = len(per_rank_functions)
    ctx = DistributedContext(use_gpu=False, groups=None)

    per_rank_outputs = [
        run_function(ctx, rank, fn, inputs, debug_mock=True)
        for rank, fn, inputs in zip(
            range(_mock_world_size), per_rank_functions, per_rank_inputs
        )
    ]
    mock_runtimes = [
        [0.0 for _ in range(num_warmup + num_repetitions)]
        for _ in range(_mock_world_size)
    ]
    return (per_rank_outputs, mock_runtimes)


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
    ctx = torch.multiprocessing.get_context("spawn")
    with ctx.Pool(world_size) as p:
        runtimes = p.starmap(per_rank_runner, enumerate(per_rank_functions))

    # Load outputs:
    per_rank_outputs = [
        torch.load(os.path.join(io_dir.name, f"out.{d}.pt")) for d in range(world_size)
    ]
    io_dir.cleanup()

    return per_rank_outputs, runtimes


def run_pytorch(
    num_devices,
    fn,
    inputs,
    use_gpu=False,
    num_repetitions=1,
    num_warmup=0,
    debug_mock=False,
):
    """Project `fn` and run on `inputs` over `num_devices` devices using the
    PyTorch backend.
    """
    # TODO check that fn uses devices [0...num_devices),
    # or run through and find max device used

    # from ..ir import cpprint
    # print(*(x.shape for x in inputs))
    # cpprint(fn)

    per_rank_fns = project(fn, tuple(v.type for v in fn.inputs), num_devices)

    per_rank_inputs = [[] for _ in range(num_devices)]
    for v, a in zip(fn.inputs, inputs):
        per_rank_inputs[v.type.device.device_id - 1].append(a)
    # for xs, per_rank_fn in zip(per_rank_inputs, per_rank_fns):
    #     print(*(x.shape for x in xs))
    #     cpprint(per_rank_fn)

    if debug_mock:
        return run_mock_multiprocess(per_rank_fns, per_rank_inputs)
    else:
        return run_multiprocesses(
            per_rank_fns,
            per_rank_inputs,
            use_gpu=use_gpu,
            num_repetitions=num_repetitions,
            num_warmup=num_warmup,
        )
