import itertools
import json
from functools import partial
import numpy as np
from operator import getitem
import os
import sys
import time
from traceback import print_exc
from typing import Any, Dict, Iterable, List, NamedTuple, Sequence, Tuple
from warnings import warn

import torch
import torch.distributed as dist
from torch import fx

from ..executor.rank_projector import project
from ..ir import Function, cpprint
from ..ir.device import Device
from ..ir.type import Int32, Int64, Float16, Float32, Type

# NOTE: The code currently suffers from this issue, more investigation needed:
# https://github.com/pytorch/pytorch/issues/11201
torch.multiprocessing.set_sharing_strategy("file_system")

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
    # Debug flag
    debug_stacktrace=bool,
    # Profile flag
    profile=bool,
    # Event trace
    trace=list,
    # Memory tracking flag
    measure_peak_memory=bool,
)

MemoryUsage = NamedTuple(
    "MemoryUsage",
    total=int,
    reserved=int,
    allocated=int,
)

BackendResults = NamedTuple(
    "BackendResults",
    per_rank_outputs=list,
    latency=float,
    peak_memory=int,
)

# TODO organize by category


def _add(x, y, ctx=None):
    return torch.add(x, y)


# TODO kwargs of these functions are required, enforce this somewhere
def _allgather(x_i, axis=0, group=None, ctx=None):
    xs = [torch.zeros_like(x_i) for _ in range(len(group))]
    if ctx.use_gpu:
        xs = [x.cuda(dist.get_rank()) for x in xs]

    dist.all_gather(xs, x_i, group=ctx.groups[group])
    x = torch.cat(xs, dim=axis)
    return x


def _allreduce(x, group=None, ctx=None):
    dist.all_reduce(x, group=ctx.groups[group])
    return x


def _cast(x, to, ctx=None):
    if to == 1:
        return x.float32()
    elif to == 6:
        return x.int32()
    elif to == 7:
        return x.long()
    elif to == 9:
        return x.bool()
    else:
        raise NotImplementedError()


def _concat(*args, axis=None, ctx=None):
    return torch.cat(args, dim=axis)


def _constant(value, device=None, ctx=None):
    output = torch.tensor(value)
    if output.shape == (1,):
        return output[0]
    if ctx.use_gpu:
        return output.cuda(dist.get_rank())
    return output


def _constant_of_shape(x, value=0, ctx=None):
    # TODO: Check if value is a single value or array?
    output = torch.full(tuple(x.int().cpu().numpy()), value[0])
    if ctx.use_gpu:
        return output.cuda(dist.get_rank())
    else:
        return output


def _div(x, y, ctx=None):
    return torch.div(x, y)


def _gather(x, y, axis=0, ctx=None):
    # TODO: Find the best Torch equivalent for this
    # torch.gather and torch.index_select do not work
    output = torch.tensor(np.take(x.cpu().numpy(), y.cpu().numpy(), axis=axis))
    # output = torch.gather(x, index=torch.LongTensor(y), dim=axis)
    if output.shape == (1,):
        return output[0]
    if ctx.use_gpu:
        return output.cuda(dist.get_rank())
    return output


def _gemm(x, y, z, alpha, beta, transA=0, transB=0, ctx=None):
    if transA:
        x = x.transpose()
    if transB:
        y = y.transpose()
    return torch.matmul(alpha * x, beta * y) + z


def _identity(x, ctx=None):
    return x


def _loss(x, y, n, ctx=None):
    return torch.square(x - y) / n


def _loss_grad(x, y, n, ctx=None):
    return 2 * (x - y) / n


def _matmul(x, y, ctx=None):
    return torch.matmul(x, y)


def _matmul_grad(x, y, dz, ctx=None):
    return (torch.matmul(dz, y.T), torch.matmul(x.T, dz))


def _mul(x, y, ctx=None):
    return torch.mul(x, y)


def _nonzero(x, ctx=None):
    # Torch nonzero returns a shape of (n, 1) instead of (1, n)
    return torch.nonzero(x).transpose(1, 0)


def _pow(x, y, ctx=None):
    return torch.pow(x, y)


def _reduce_mean(x, axes, keepdims=1, ctx=None):
    return torch.mean(x, dim=axes, keepdim=bool(keepdims))


def _reshape(x, y, ctx=None):
    new_shape = tuple(int(v.item()) for v in list(y))
    return torch.reshape(x, new_shape)


def _recv(shape=None, from_d=None, group=None, dtype=None, ctx=None):
    if isinstance(dtype, Int32):
        x = torch.zeros(shape).int()
    elif isinstance(dtype, Int64):
        x = torch.zeros(shape).long()
    elif isinstance(dtype, Float16):
        x = torch.zeros(shape).half()
    elif isinstance(dtype, Float32):
        x = torch.zeros(shape).float()
    else:
        raise NotImplementedError(dtype)

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


def _sgd(*xs, lr=None, ctx=None):
    weights = xs[: (len(xs) // 2)]
    gradients = xs[(len(xs) // 2) :]
    updated_weights = []
    for w, dw in zip(weights, gradients):
        dw *= lr
        w -= dw
        updated_weights.append(w)
    return tuple(updated_weights)


def _shape(x, ctx=None):
    output = torch.tensor(x.shape)
    if ctx.use_gpu:
        return output.cuda(dist.get_rank())
    return output


def _slice(x, starts, ends, axes, steps=None, ctx=None):
    # TODO: Find the best PyTorch equivalent for this
    starts = [v.item() for v in list(starts)]
    ends = [v.item() for v in list(ends)]
    axes = [v.item() for v in list(axes)]
    if steps is None:
        steps = [1] * len(starts)
    elif steps.shape == ():
        steps = [steps.item()] * len(starts)
    else:
        assert len(steps) == len(starts)
    slices = {
        axis: slice(s, e, step) for (s, e, axis, step) in zip(starts, ends, axes, steps)
    }
    slices = tuple(slices.get(d, slice(None)) for d in range(x.ndim))
    return x[slices]


def _softmax(x, axis, ctx=None):
    return torch.nn.functional.softmax(x, dim=axis)


def _split(x, axis, split, ctx=None):
    return torch.split(x, split, axis)


def _sqrt(x, ctx=None):
    return torch.sqrt(x)


def _sub(x, y, ctx=None):
    return torch.sub(x, y)


def _squeeze(x, axes=None, ctx=None):
    if axes:
        return torch.squeeze(x, dim=axes[0])
    else:
        return torch.squeeze(x)


def _tanh(x, ctx=None):
    return torch.tanh(x)


def _transpose(x, perm, ctx=None):
    return x.permute(perm)


def _unsqueeze(x, axes, ctx=None):
    for dim in axes[::-1]:
        x = torch.unsqueeze(x, dim=dim)
    return x


_op_to_torch = {
    "Add": torch.add,
    "Cast": _cast,
    "Add": _add,
    "Concat": _concat,
    "Constant": _constant,
    "ConstantOfShape": _constant_of_shape,
    "Div": _div,
    "Gather": _gather,
    "Gemm": _gemm,
    "Identity": _identity,
    "Loss": _loss,
    "LossGrad": _loss_grad,
    "MatMul": _matmul,
    "MatMulGrad": _matmul_grad,
    "MPIAllgather": _allgather,
    "MPIAllreduce": _allreduce,
    "Mul": _mul,
    "NonZero": _nonzero,
    "Pow": _pow,
    "RecvP2P": _recv,
    "ReduceMean": _reduce_mean,
    "Relu": _relu,
    "ReluGrad": _relu_grad,
    "Reshape": _reshape,
    "SendP2P": _send,
    "SGDOptimizer": _sgd,
    "Shape": _shape,
    "Slice": _slice,
    "Softmax": _softmax,
    "Split": _split,
    "Sqrt": _sqrt,
    "Squeeze": _squeeze,
    "Sub": _sub,
    "Tanh": _tanh,
    "Transpose": _transpose,
    "Unsqueeze": _unsqueeze,
    # TODO rename MPI<opname> to Comm<opname> or Dist<opname>
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


def _mock_recv(shape=None, device=None, dtype=None, ctx=None):
    if isinstance(dtype, Int64):
        x = torch.zeros(shape).long()
    elif isinstance(dtype, Float32):
        x = torch.zeros(shape).float()
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


def get_memory_usage(rank, verbose=False):
    t = torch.cuda.get_device_properties(rank).total_memory
    r = torch.cuda.memory_reserved(rank)
    a = torch.cuda.memory_allocated(rank)
    if verbose:
        print(f"[Rank {rank}] Total: {t} Reserved: {r} Allocated: {a} Free: {r-a}")
    return MemoryUsage(t, r, a)


def run_function(
    ctx: DistributedContext,
    fn: Function,
    inputs: List[Any],
    rank: int,
    debug_mock=False,
    op_runtimes_ts: float = None,
):
    """Runs DistIR Function `fn` on `inputs` in a distributed context `ctx` by
    converting each DistIR op to its torch implementation as given in _op_to_torch.
    """
    record_op_runtimes = op_runtimes_ts is not None
    op_to_torch = _mock_op_to_torch if debug_mock else _op_to_torch
    value_map = {}

    # Add inputs to value_map
    for v, x in zip(fn.inputs, inputs):
        value_map[v] = x
    assert len(fn.inputs) == len(inputs)

    if ctx.measure_peak_memory:
        torch.cuda.synchronize(rank)
        memory_usage = get_memory_usage(rank)
        allocated_memory = memory_usage.allocated
        peak_memory = allocated_memory
    else:
        peak_memory = 0.0

    if record_op_runtimes:
        op_runtimes = []

    # Run ops
    for op in fn.ops:
        inputs = tuple(value_map[v] for v in op.inputs)
        kwargs = {} if op.attributes is None else {**op.attributes}
        kwargs["ctx"] = ctx

        if record_op_runtimes:
            start = time.time()
        output = op_to_torch[op.op_type](*inputs, **kwargs)
        if record_op_runtimes:
            if ctx.use_gpu:
                torch.cuda.synchronize()
            end = time.time()
            op_runtimes.append(end - start)

        if len(op.outputs) > 1:
            assert isinstance(output, tuple)
            for i, v in enumerate(op.outputs):
                value_map[v] = output[i]
                # TODO: Hide this under debug flag
                # if torch.any(torch.isnan(output[i])):
                #     warn(f"NaNs in op {op} output {i}")
        elif len(op.outputs) == 1:
            value_map[op.outputs[0]] = output
            # TODO: Hide this under debug flag
            # if torch.any(torch.isnan(output)):
            #    warn(f"NaNs in op {op.name} output {0}")

        if ctx.measure_peak_memory:
            torch.cuda.synchronize(rank)
            memory_usage = get_memory_usage(rank)
            peak_memory = max(peak_memory, memory_usage.allocated)

        # Free tensors that are not used again
        for v in op.inputs:
            if v in value_map and fn.last_use(v) == op and not (v in fn.outputs):
                del value_map[v]

    if record_op_runtimes:
        trace = []
        ts = op_runtimes_ts
        assert len(fn.ops) == len(op_runtimes)
        for op, runtime in zip(fn.ops, op_runtimes):
            trace.append(
                {
                    "name": op.op_type,
                    "ph": "X",
                    "ts": ts,
                    "dur": runtime * 1e6,
                    "pid": 0,
                    "tid": rank,
                }
            )
            ts += runtime * 1e6
        ctx.trace[rank] += trace

    # Return outputs
    return tuple(value_map[v] for v in fn.outputs), peak_memory


def run_process(ctx, num_warmup_steps, num_repetitions, rank, fn, inputs):
    """The Python function on rank `rank` that runs DistIR function `fn` on
    (torch) inputs `inputs`. The function is run
    `num_warmup_steps + num_repetitions` times. The outputs of the last run are
    returned, along with the last `num_repetitions` runtimes.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # TODO make these configurable
    os.environ["MASTER_PORT"] = "29500"
    backend = "nccl" if ctx.use_gpu else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=ctx.world_size)

    # Create the process groups used by fn's communication ops
    for group in ctx.groups_list:
        ranks = [ctx.device_to_rank[d] for d in group]
        # ctx is a curried arg, hence is thread-local and can be modified:
        ctx.groups[group] = dist.new_group(ranks)

    if ctx.use_gpu:
        # Move inputs to GPU
        print(f"Rank {rank}: Moving inputs to GPU...")
        gpu_inputs = []
        torch.cuda.synchronize(rank)
        memory_usage = get_memory_usage(rank)
        print(
            f"Rank {rank}: reserved={memory_usage.reserved}, "
            f"allocated={memory_usage.allocated}"
        )
        for i, t in enumerate(inputs):
            input_size = t.nelement() * t.element_size()
            print(f"Rank {rank}: Moving input {i} of {input_size} bytes to GPU...")
            gpu_inputs.append(t.cuda(rank))
            torch.cuda.synchronize(rank)
            memory_usage = get_memory_usage(rank)
            print(
                f"Rank {rank}: reserved={memory_usage.reserved}, "
                f"allocated={memory_usage.allocated}"
            )
        inputs = gpu_inputs

    events = []

    def add_event():
        if ctx.use_gpu:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        else:
            events.append(time.perf_counter())

    if ctx.debug_stacktrace:
        try:
            outputs, peak_memory = run_function(ctx, fn, inputs, rank)
            if ctx.world_size > 1:
                torch.distributed.barrier()
        except Exception as e:
            print_exc()
        print(f"{rank}: PyTorch backend exiting after 1 run in debug mode.")
        dist.destroy_process_group()
        return BackendResults(None, None, None)

    def run(p=None):
        op_runtimes_ts = None
        for i in range(num_warmup_steps + num_repetitions):
            record_op_runtimes = ctx.profile and i >= num_warmup_steps
            if record_op_runtimes and op_runtimes_ts is None:
                op_runtimes_ts = 0.0
            # TODO: Handle failures here?
            print(f"Rank {rank}: Running step {i}...")
            add_event()
            outputs, peak_memory = run_function(
                ctx, fn, inputs, rank, op_runtimes_ts=op_runtimes_ts
            )
            if i == (num_warmup_steps + num_repetitions - 1):
                add_event()
            if p is not None:
                p.step()
            if record_op_runtimes:
                op_runtimes_ts = max(
                    ctx.trace[rank][-1]["ts"] + ctx.trace[rank][-1]["dur"]
                    for rank in ctx.trace.keys()
                )
        return outputs, peak_memory

    if ctx.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0, warmup=num_warmup_steps, active=num_repetitions
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"{fn.name}_profile"
            ),
        ) as p:
            outputs, peak_memory = run(p)
    else:
        outputs, peak_memory = run()

    if ctx.use_gpu:
        # Move outputs back to cpu
        try:
            outputs = [t.cpu() for t in outputs]
        except Exception as e:
            print(outputs)
            outputs = None
        torch.cuda.synchronize()
        runtimes = [
            events[i].elapsed_time(events[i + 1]) / 1e3 for i in range(len(events) - 1)
        ]
    else:
        runtimes = [events[i + 1] - events[i] for i in range(len(events) - 1)]

    dist.destroy_process_group()
    return outputs, runtimes[num_warmup_steps:], peak_memory


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
        run_function(ctx, fn, inputs, debug_mock=True)[0]
        for rank, fn, inputs in zip(
            range(_mock_world_size), per_rank_functions, per_rank_inputs
        )
    ]
    mock_latency = 0.0
    mock_peak_memory = 0.0
    return BackendResults(per_rank_outputs, mock_latency, mock_peak_memory)


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

    if ctx.debug_stacktrace:
        sys.exit(1)

    if ctx.profile:
        trace = list(itertools.chain.from_iterable(list(ctx.trace.values())))
        with open(f"{per_rank_functions[0].name}_profile/trace.json", "w") as f:
            json.dump(trace, f)

    per_rank_outputs, per_rank_runtimes, per_rank_peak_memory = zip(*outputs)
    per_rank_runtimes = np.array(per_rank_runtimes)
    latency = np.median(
        [np.max(per_rank_runtimes[:, i]) for i in range(len(per_rank_runtimes[0]))]
    )
    peak_memory = np.max(per_rank_peak_memory)
    return BackendResults(per_rank_outputs, latency, peak_memory)


def run_pytorch(
    fn: Function,
    inputs: Tuple[Any],
    input_types: Tuple[Type] = None,
    use_gpu=False,
    num_repetitions=1,
    num_warmup=0,
    debug_mock=False,
    debug_stacktrace=False,
    profile=False,
    measure_peak_memory=False,
):
    """Project `fn` and run on `inputs` over `num_devices` devices using the
    PyTorch backend.

    `inputs` is a list/tuple of the same length as `fn.inputs`.  `input_types`
    is a list/tuple of abstract/concrete inputs used for projection.

    The run is repeated 'num_warmup + num_repetitions` times, and runtimes from
    the last `num_repetitions` runs are returned along with the outputs of the
    last run.

    `debug_mock` runs the function sequentially, replacing communication ops with
    mock versions that return arbitrary values. `debug_stacktrace` wraps the
    run function with a try-catch block and prints the stack trace and exits if
    any thread raises an exception. `profile` runs the code with the PyTorch
    profiler and outputs logs to TensorBoard. `measure_peak_memory` keeps track
    of the peak memory usage reported by PyTorch (note that this requires
    synchronizing after every op, which will result in runtime overhead).
    """

    if measure_peak_memory:
        assert use_gpu

    if input_types is None:
        input_types = tuple(v.type for v in fn.inputs)
    else:
        assert len(input_types) == len(fn.inputs)

    print("Projecting function...")
    device_to_fns, groups = project(fn, input_types)

    # Map between DistIR devices and pytorch ranks:
    device_to_rank = {}
    world_size = 0
    per_rank_fns = []
    for d in device_to_fns:
        device_to_rank[d] = world_size
        per_rank_fns.append(device_to_fns[d])
        world_size += 1

    if profile:
        manager = torch.multiprocessing.Manager()
        trace = manager.dict()
        for d in sorted(device_to_rank.keys()):
            trace[device_to_rank[d]] = []
    else:
        trace = None

    ctx = DistributedContext(
        world_size=world_size,
        use_gpu=use_gpu,
        groups={},
        groups_list=list(groups),
        device_to_rank=device_to_rank,
        debug_stacktrace=debug_stacktrace,
        profile=profile,
        trace=trace,
        measure_peak_memory=measure_peak_memory,
    )

    per_rank_inputs = [[] for _ in range(world_size)]
    for t, a in zip(input_types, inputs):
        per_rank_inputs[device_to_rank[t.device]].append(a)
    assert len(fn.inputs) == len(inputs)

    print("Launching distributed processes...")
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
