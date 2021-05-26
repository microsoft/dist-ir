from functools import partial
from operator import getitem
import os
import sys
from time import perf_counter
from traceback import print_exc
from typing import Any, Dict, Iterable, List, NamedTuple, Tuple

import torch
import torch.distributed as dist
from torch import fx

from ..executor.rank_projector import project
from ..ir import Function, cpprint, pformat
from ..ir.device import Device


DistributedContext = NamedTuple(
    "DistributedContext",
    world_size=int,
    use_gpu=bool,
    # Map from DistIR device to PyTorch backend rank
    device_to_rank=Dict[Device, int],
    # Maps tuple of ranks to ProcessGroup
    groups=Dict[Tuple[int], Any],
    # Temp store of group IDs until threads can create ProcessGroups
    groups_list=Iterable[Tuple[int]],
)


# TODO organize by category


def _add(x, y, ctx=None):
    return torch.add(x, y)


# TODO kwargs of these functions are required, enforce this somewhere
def _allgather(x_i, dim=0, group=None, ctx=None):
    xs = [torch.zeros_like(x_i) for _ in range(len(group))]
    if ctx.use_gpu:
        xs = [x.cuda(dist.get_rank()) for x in xs]

    dist.all_gather(xs, x_i, group=ctx.groups[group])
    x = torch.cat(xs, dim=dim)
    return x


def _allreduce(x, group=None, ctx=None):
    dist.all_reduce(x, group=ctx.groups[group])
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


def _recv(shape=None, from_d=None, group=None, ctx=None):
    x = torch.zeros(shape)
    src_rank = ctx.device_to_rank[from_d]
    if ctx.use_gpu:
        x = x.cuda(dist.get_rank())
        dist.broadcast(x, src_rank, group=ctx.groups[group])
    else:
        dist.recv(x, src_rank)
    return x


def _relu(x, ctx=None):
    return torch.relu(x)


def _relu_grad(x, dy, ctx=None):
    dx = dy.clone()
    dx[x <= 0] = 0
    return dx


def _send(x, to_d=None, group=None, ctx=None):
    if ctx.use_gpu:
        src_rank = dist.get_rank()
        dist.broadcast(x, src_rank, group=ctx.groups[group])
    else:
        dst_rank = ctx.device_to_rank[to_d]
        dist.send(x, dst_rank)
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
    """Deprecated. Converts a DistIR Function to a PyTorch nn.Module using
    torch.fx.
    """
    g = fx.Graph()
    value_map = {}

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
    fn: Function,
    inputs: List[Any],
    debug_mock=False,
):
    """Runs DistIR Function `fn` on `inputs` in a distributed context `ctx` by
    converting each DistIR op to its torch implementation as given in _op_to_torch.
    """
    op_to_torch = _mock_op_to_torch if debug_mock else _op_to_torch
    value_map = {}

    # Add inputs to value_map
    for v, x in zip(fn.inputs, inputs):
        value_map[v] = x
    assert len(fn.inputs) == len(inputs)

    # Run ops
    for op in fn.ops:
        # op_str = pformat(op).replace("\n", " ")
        # print(f"{rank}: {op_str}")
        # sys.stdout.flush()
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

        # Free tensors that are not used again
        for v in op.inputs:
            if v in value_map and fn.last_use(v) == op and not (v in fn.outputs):
                del value_map[v]

        # print(f"{rank}: {op_str}")
        # sys.stdout.flush()

    # Return outputs
    return tuple(value_map[v] for v in fn.outputs)


def run_process(ctx, num_warmup_steps, num_repetitions, rank, fn, inputs):
    """The Python function on rank `rank` that runs DistIR function `fn` on
    (torch) inputs `inputs`. The function is run
    `num_warmup_steps + num_repetitions` times. The outputs of the last run are
    returned, along with the last `num_repetitions` runtimes.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    backend = "nccl" if ctx.use_gpu else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=ctx.world_size)

    # Create the process groups used by fn's communication ops
    for group in ctx.groups_list:
        ranks = [ctx.device_to_rank[d] for d in group]
        # TODO ctx is copied or shared among threads?
        ctx.groups[group] = dist.new_group(ranks)

    if ctx.use_gpu:
        # Move module and inputs to GPU
        # TODO check if interpreted code is running on GPU (check all inputs?)
        # module = module.cuda(rank)
        inputs = [t.cuda(rank) for t in inputs]

    events = []

    def add_event():
        if ctx.use_gpu:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        else:
            events.append(perf_counter())

    # Time a bunch of executions, then execute once for output values
    add_event()
    for _ in range(num_warmup_steps + num_repetitions):
        # res = module(*inputs)
        # try:
        #     outputs = run_function(ctx, fn, inputs)
        # except Exception as e:
        #     print_exc()
        #     sys.exit(1)
        outputs = run_function(ctx, fn, inputs)
        if ctx.world_size > 1:
            torch.distributed.barrier()
        add_event()

    if ctx.use_gpu:
        # Move outputs back to cpu
        outputs = [t.cpu() for t in outputs]

    if ctx.use_gpu:
        torch.cuda.synchronize()
        runtimes = [
            events[i].elapsed_time(events[i + 1]) / 1e3 for i in range(len(events) - 1)
        ]
    else:
        runtimes = [events[i + 1] - events[i] for i in range(len(events) - 1)]

    dist.destroy_process_group()
    return outputs, runtimes[num_warmup_steps:]


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
        run_function(ctx, fn, inputs, debug_mock=True)
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
    ctx,
    per_rank_functions: Tuple[Function],
    per_rank_inputs: Tuple[Any],
    num_repetitions=1,
    num_warmup=0,
):
    assert len(per_rank_functions) == len(per_rank_inputs)
    args = [
        (r, f, x) for (r, (f, x)) in enumerate(zip(per_rank_functions, per_rank_inputs))
    ]

    per_rank_runner = partial(run_process, ctx, num_warmup, num_repetitions)
    mp = torch.multiprocessing.get_context("spawn")
    with mp.Pool(ctx.world_size) as p:
        outputs = p.starmap(per_rank_runner, args)

    per_rank_outputs, runtimes = zip(*outputs)
    return per_rank_outputs, runtimes


def run_pytorch(
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
    # print(*(x.shape for x in inputs))
    # cpprint(fn)

    device_to_fns, groups = project(fn, tuple(v.type for v in fn.inputs))

    # Map between DistIR devices and pytorch ranks:
    device_to_rank = {}
    world_size = 0
    per_rank_fns = []
    for d in device_to_fns:
        device_to_rank[d] = world_size
        per_rank_fns.append(device_to_fns[d])
        world_size += 1

    ctx = DistributedContext(
        world_size=world_size,
        use_gpu=use_gpu,
        groups={},
        groups_list=list(groups),
        device_to_rank=device_to_rank,
    )

    per_rank_inputs = [[] for _ in range(world_size)]
    for v, a in zip(fn.inputs, inputs):
        per_rank_inputs[device_to_rank[v.type.device]].append(a)

    # for xs, per_rank_fn in zip(per_rank_inputs, per_rank_fns):
    #     print(*(x.shape for x in xs))
    #     cpprint(per_rank_fn)

    if debug_mock:
        return run_mock_multiprocess(per_rank_fns, per_rank_inputs)
    else:
        return run_multiprocesses(
            ctx,
            per_rank_fns,
            per_rank_inputs,
            num_repetitions=num_repetitions,
            num_warmup=num_warmup,
        )
