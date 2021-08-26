"""
This file defines a register of reference implementations for communication ops.
They work explicitly on ConcreteValues and return ConcreteValues on the
appropriate devices. (This is why they cannot be wrapped like the numpy register.)
"""

import numpy as np

from .concrete_value import ConcreteValue
from .numpy_register import identity, split_uniform


def mpi_allgather(op, *xs):
    dim = op.attributes["axis"]
    v = np.concatenate(tuple(x.val for x in xs), axis=dim)
    return tuple(ConcreteValue(v, x.device) for x in xs)


def mpi_allreduce(op, *xs):
    sum_ = np.sum((x.val for x in xs), axis=0)
    return tuple(ConcreteValue(sum_, x.device) for x in xs)


def mpi_broadcast(op, x):
    return tuple(ConcreteValue(x.val, device) for device in op.attributes["devices"])


def mpi_gather(op, *xs):
    dim = op.attributes["axis"]
    v = np.concatenate(tuple(x.val for x in xs), axis=dim)
    return ConcreteValue(v, op.attributes["device"])


def mpi_reduce(op, *xs):
    v = np.sum((x.val for x in xs), axis=0)
    return ConcreteValue(v, op.attributes["device"])


def mpi_scatter(op, x):
    dim = op.attributes["axis"]
    num_splits = len(op.attributes["devices"])
    return tuple(
        ConcreteValue(y, device)
        for y, device in zip(
            np.split(x.val, num_splits, axis=dim), op.attributes["devices"]
        )
    )


def send(op, x):
    return ConcreteValue(x.val, op.attributes["device"])


CommunicationRegister = {
    # (
    #    "MPIAllreduceFromTupleType",
    #    (tuple,),
    # ): lambda op, *xs: mpi_allreduce(op, *xs[0]),
    ("MPIAllgather", (ConcreteValue,) * 2): mpi_allgather,
    ("MPIAllgather", (ConcreteValue,) * 4): mpi_allgather,
    ("MPIAllgather", (ConcreteValue,) * 8): mpi_allgather,
    ("MPIAllgather", (ConcreteValue,) * 16): mpi_allgather,
    ("MPIAllgather", (ConcreteValue,) * 32): mpi_allgather,
    ("MPIAllgather", (ConcreteValue,) * 64): mpi_allgather,
    ("MPIAllgather", (ConcreteValue,) * 128): mpi_allgather,
    ("MPIAllgather", (ConcreteValue,) * 256): mpi_allgather,
    ("MPIAllgather", (ConcreteValue,) * 512): mpi_allgather,
    ("MPIAllgather", (ConcreteValue,) * 1024): mpi_allgather,
    ("MPIAllgather", (ConcreteValue,) * 2048): mpi_allgather,
    ("MPIAllgather", (ConcreteValue,) * 4096): mpi_allgather,
    ("MPIAllgather", (ConcreteValue,) * 8192): mpi_allgather,
    ("MPIAllreduce", (ConcreteValue,) * 2): mpi_allreduce,
    ("MPIAllreduce", (ConcreteValue,) * 4): mpi_allreduce,
    ("MPIAllreduce", (ConcreteValue,) * 8): mpi_allreduce,
    ("MPIAllreduce", (ConcreteValue,) * 16): mpi_allreduce,
    ("MPIAllreduce", (ConcreteValue,) * 32): mpi_allreduce,
    ("MPIAllreduce", (ConcreteValue,) * 64): mpi_allreduce,
    ("MPIAllreduce", (ConcreteValue,) * 128): mpi_allreduce,
    ("MPIAllreduce", (ConcreteValue,) * 256): mpi_allreduce,
    ("MPIAllreduce", (ConcreteValue,) * 512): mpi_allreduce,
    ("MPIAllreduce", (ConcreteValue,) * 1024): mpi_allreduce,
    ("MPIAllreduce", (ConcreteValue,) * 2048): mpi_allreduce,
    ("MPIAllreduce", (ConcreteValue,) * 4096): mpi_allreduce,
    ("MPIAllreduce", (ConcreteValue,) * 8192): mpi_allreduce,
    ("MPIBroadcast", (ConcreteValue,)): mpi_broadcast,
    ("MPIBroadcastToTupleType", (ConcreteValue,)): mpi_broadcast,
    ("MPIGather", (ConcreteValue,) * 2): mpi_gather,
    ("MPIGather", (ConcreteValue,) * 4): mpi_gather,
    ("MPIGather", (ConcreteValue,) * 8): mpi_gather,
    ("MPIGather", (ConcreteValue,) * 16): mpi_gather,
    ("MPIGather", (ConcreteValue,) * 32): mpi_gather,
    ("MPIGather", (ConcreteValue,) * 64): mpi_gather,
    ("MPIGather", (ConcreteValue,) * 128): mpi_gather,
    ("MPIGather", (ConcreteValue,) * 256): mpi_gather,
    ("MPIGather", (ConcreteValue,) * 512): mpi_gather,
    ("MPIGather", (ConcreteValue,) * 1024): mpi_gather,
    ("MPIGather", (ConcreteValue,) * 2048): mpi_gather,
    ("MPIGather", (ConcreteValue,) * 4096): mpi_gather,
    ("MPIGather", (ConcreteValue,) * 8192): mpi_gather,
    # ("MPIGatherFromTupleType", (tuple,)): lambda op, *xs: mpi_gather(op, *xs[0]),
    ("MPIReduce", (ConcreteValue,) * 2): mpi_reduce,
    ("MPIReduce", (ConcreteValue,) * 4): mpi_reduce,
    ("MPIReduce", (ConcreteValue,) * 8): mpi_reduce,
    ("MPIReduce", (ConcreteValue,) * 16): mpi_reduce,
    ("MPIReduce", (ConcreteValue,) * 32): mpi_reduce,
    ("MPIReduce", (ConcreteValue,) * 64): mpi_reduce,
    ("MPIReduce", (ConcreteValue,) * 128): mpi_reduce,
    ("MPIReduce", (ConcreteValue,) * 256): mpi_reduce,
    ("MPIReduce", (ConcreteValue,) * 512): mpi_reduce,
    ("MPIReduce", (ConcreteValue,) * 1024): mpi_reduce,
    ("MPIReduce", (ConcreteValue,) * 2048): mpi_reduce,
    ("MPIReduce", (ConcreteValue,) * 4096): mpi_reduce,
    ("MPIReduce", (ConcreteValue,) * 8192): mpi_reduce,
    ("MPIScatter", (ConcreteValue,)): mpi_scatter,
    ("MPIScatterToTupleType", (ConcreteValue,)): mpi_scatter,
    ("Send", (ConcreteValue,)): send,
}
