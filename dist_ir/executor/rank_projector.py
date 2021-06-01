from collections import defaultdict
from dist_ir.executor.type_inference import TypePropRegister
from typing import Any, Dict, Sequence, Set, Tuple

from ..ir import Function, FunctionMaker, Device, Op, Value
from ..ir.type import Type, Tensor
from .absint import AbstractState, AbstractInterpreter


# TODO merge this with torch backend -- it breaks semantics to have P2P send/recv


class ProjectorState(AbstractState):
    """The Abstract Interpreter state for projection. It keeps a mapping from
    Devices to per-rank Functions, and a set of Device groups that perform
    collective communication.
    """

    def __init__(self, function: Function, inputs: Sequence[Any]):
        AbstractState.__init__(self, function, inputs)
        self.per_rank_fns: Dict[Device, FunctionMaker] = defaultdict(FunctionMaker)
        self.groups: Set[Tuple[Device]] = set()


def _get_input_devices(op: Op):
    return list(set(x.type.device for x in op.inputs))


def _make_group(devices):
    """Return a hashable representation of a group of devices. This is needed by
    the backend, which maps them to process groups for communication primitives.
    """
    return tuple(sorted(set(devices)))


# TODO should projectors just get the per_rank_fns dict instead of full state?


def _identity_projector(op: Op, state: ProjectorState):
    """Projects op unchanged to its device's per-rank program.
    The inputs of op must all be on a single device.
    """
    devices = _get_input_devices(op)
    assert len(devices) == 1 and devices[0] is not None

    state.per_rank_fns[devices[0]].ops.append(op)
    # state.per_rank_fns[d].add_op(op.op_type, name=op.name, inputs=op.inputs, )


def _collective_projector(op: Op, state: ProjectorState):
    """Projects a collective op over D devices that has D inputs and D outputs,
    one on each device."""
    assert len(op.inputs) == len(op.outputs)
    group = _make_group(v.type.device for v in op.inputs + op.outputs)
    attributes = {
        **(op.attributes if op.attributes is not None else {}),
        "group": group,
    }
    for in_v, out_v in zip(op.inputs, op.outputs):
        assert in_v.type.device == out_v.type.device
        d = in_v.type.device

        new_op = Op(
            op.op_type,
            inputs=(in_v,),
            output_values=(out_v,),
            attributes=attributes,
        )
        state.per_rank_fns[d].ops.append(new_op)


def _gather_projector(op: Op, state: ProjectorState):
    devices = set(v.type.device for v in op.inputs)
    assert len(op.inputs) == len(devices)
    assert len(op.outputs) == 1 and op.outputs[0].type.device in devices
    attributes = {
        **(op.attributes if op.attributes is not None else {}),
        "group": _make_group(devices),
    }
    for in_v in op.inputs:
        d = in_v.type.device
        new_op = Op(
            op.op_type,
            inputs=(in_v,),
            output_values=op.outputs,  # TODO only on dst device!
            attributes=attributes,
        )
        state.per_rank_fns[d].ops.append(new_op)


def _send_projector(op: Op, state: ProjectorState):
    from_d = op.inputs[0].type.device
    to_d = op.attributes["device"]
    group = _make_group((from_d, to_d))
    state.per_rank_fns[from_d].ops.append(
        Op(
            "SendP2P",
            inputs=op.inputs,
            attributes={"to_d": to_d, "group": group},
        )
    )
    state.per_rank_fns[to_d].ops.append(
        Op(
            "RecvP2P",
            output_values=(op.outputs[0],),
            attributes={
                "shape": op.inputs[0].type.shape,
                "from_d": from_d,
                "group": group,
            },
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
    ("MPIAllgather", (Tensor,) * 2): _collective_projector,
    ("MPIAllgather", (Tensor,) * 4): _collective_projector,
    ("MPIAllgather", (Tensor,) * 8): _collective_projector,
    ("MPIAllgather", (Tensor,) * 16): _collective_projector,
    ("MPIAllreduce", (Tensor,) * 2): _collective_projector,
    ("MPIAllreduce", (Tensor,) * 4): _collective_projector,
    ("MPIAllreduce", (Tensor,) * 8): _collective_projector,
    ("MPIAllreduce", (Tensor,) * 16): _collective_projector,
    ("MPIGather", (Tensor,) * 2): _gather_projector,
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

            # If op involves more than one device, create a group
            devices = [v.device for v in outputs] + [v.type.device for v in op.inputs]
            group = _make_group(devices)
            if len(group) > 1:
                state.groups.add(group)

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


def project(
    fn: Function, input_types: Sequence[Type]
) -> Tuple[Dict[Device, Function], Set[Tuple[Device]]]:
    """Project `fn` to per-rank functions. Returns a mapping from Devices to
    per-rank Functions, and a set of Device groups that perform collective
    communications in `fn`.
    """
    state = ProjectorState(fn, input_types)

    # Project fn's inputs to each per-rank fn:
    for v in fn.inputs:
        state.per_rank_fns[v.type.device].inputs.append(v)

    state = Projector.interpret(fn, input_types, state=state)

    result_fns = {}
    for d, per_rank_fn in state.per_rank_fns.items():
        result_fns[d] = per_rank_fn.finalize()

    return result_fns, state.groups
