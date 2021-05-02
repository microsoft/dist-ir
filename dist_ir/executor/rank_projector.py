from collections import defaultdict
from dist_ir.executor.type_inference import TypePropRegister
from typing import Any, Dict, Sequence

from ..ir import Function, FunctionMaker, Device, Op
from ..ir.type import Type, Tensor
from .absint import AbstractState, AbstractInterpreter


# TODO merge this with torch backend -- it breaks semantics to have P2P send/recv


class ProjectorState(AbstractState):
    def __init__(self, function: Function, inputs: Sequence[Any]):
        AbstractState.__init__(self, function, inputs)
        self.per_rank_fns: Dict[Device, FunctionMaker] = defaultdict(FunctionMaker)


def _get_input_devices(op: Op):
    return list(set(x.type.device for x in op.inputs))


# TODO should projectors just get the per_rank_fns dict instead of full state?


def _identity_projector(op: Op, state: ProjectorState):
    """Projects op unchanged to its device's per-rank program.
    The inputs of op must all be on a single device.
    """
    devices = _get_input_devices(op)
    assert len(devices) == 1 and devices[0] is not None

    state.per_rank_fns[devices[0]].ops.append(op)
    # state.per_rank_fns[d].add_op(op.op_type, name=op.name, inputs=op.inputs, )


def _mpi_allgather_projector(op: Op, state: ProjectorState):
    assert len(op.inputs) == len(op.outputs)
    for in_v, out_v in zip(op.inputs, op.outputs):
        assert in_v.type.device == out_v.type.device
        d = in_v.type.device

        new_op = Op(
            "MPIAllgather",
            inputs=(in_v,),
            output_values=(out_v,),
            attributes=op.attributes,
        )
        state.per_rank_fns[d].ops.append(new_op)


def _send_projector(op: Op, state: ProjectorState):
    from_d = op.inputs[0].type.device
    to_d = op.attributes["device"]
    state.per_rank_fns[from_d].ops.append(
        Op("SendP2P", inputs=op.inputs, attributes={"device": to_d.device_id})
    )
    state.per_rank_fns[to_d].ops.append(
        Op(
            "RecvP2P",
            output_values=(op.outputs[0],),
            attributes={"shape": op.inputs[0].type.shape, "device": from_d.device_id},
        )
    )


ProjectorRegister = {
    ("Add", (Tensor, Tensor)): _identity_projector,
    ("Concat", (Tensor, Tensor)): _identity_projector,
    ("Identity", (Tensor,)): _identity_projector,
    ("Loss", (Tensor, Tensor)): _identity_projector,
    ("LossGrad", (Tensor, Tensor)): _identity_projector,
    ("MatMul", (Tensor, Tensor)): _identity_projector,
    ("MatMulGrad", (Tensor, Tensor, Tensor)): _identity_projector,
    ("MPIAllgather", (Tensor,) * 2): _mpi_allgather_projector,
    ("MPIAllgather", (Tensor,) * 4): _mpi_allgather_projector,
    ("MPIAllgather", (Tensor,) * 8): _mpi_allgather_projector,
    ("MPIAllgather", (Tensor,) * 16): _mpi_allgather_projector,
    ("Relu", (Tensor,)): _identity_projector,
    ("ReluGrad", (Tensor, Tensor)): _identity_projector,
    ("Send", (Tensor,)): _send_projector,
}


def _create_semantics(type_prop_register, projector_register):
    """Creates a semantics for AbstractInterpreter by combining a register of
    projector functions and the type propagation register.
    """

    def convert_impl(type_prop_fn, projector):
        def semantics(op: Op, state: AbstractState):
            # Find the op's inputs in state's environment
            inputs = tuple(state.env[v] for v in op.inputs)
            # Run the type propagation function
            outputs = type_prop_fn(op, *inputs)

            # Write outputs to state's environment
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for x, val in zip(op.outputs, outputs):
                state.env[x] = val

            # Project op and add to appropriate per-rank function
            projector(op, state)

        return semantics

    signatures = set(projector_register.keys()).intersection(type_prop_register.keys())

    return {
        f: convert_impl(type_prop_register[f], projector_register[f])
        for f in signatures
    }


Projector = AbstractInterpreter(
    AbstractState=ProjectorState,
    semantics=_create_semantics(TypePropRegister, ProjectorRegister),
)


def project(fn: Function, input_types: Sequence[Type]):
    """Project fn to a sequence of per-rank functions."""
    state = ProjectorState(fn, input_types)

    # Project fn's inputs to each per-rank fn:
    for v in fn.inputs:
        state.per_rank_fns[v.type.device].inputs.append(v)

    state = Projector.interpret(fn, input_types, state=state)

    return {d: state.per_rank_fns[d].finalize() for d in state.per_rank_fns}
