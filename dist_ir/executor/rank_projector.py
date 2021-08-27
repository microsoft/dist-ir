from collections import defaultdict
from typing import Any, Dict, Sequence, Set, Tuple

from ..ir import Function, FunctionMaker, Device, Op
from ..ir.type import Type, Float32, Float64, Int64, Tensor
from .absint import (
    AbstractState,
    dispatch,
    interpreter,
    update_semantics_with_register,
)


# TODO merge this with torch backend -- it breaks semantics to have P2P send/recv


# TODO should projectors just get the function instead of full state?


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
    return list(set(x.type.device for x in op.inputs if x.type.device is not None))


def _make_group(devices):
    """Return a hashable representation of a group of devices. This is needed by
    the backend, which maps them to process groups for communication primitives.
    """
    return tuple(sorted(set(devices)))


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


def _constant_projector(op: Op, state: ProjectorState):
    assert len(op.outputs) == 1
    device = op.attributes["device"]
    state.per_rank_fns[device].ops.append(op)


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


def _identity_projector(op: Op, state: ProjectorState):
    """Projects op unchanged to its device's per-rank program.
    The inputs of op must all be on a single device.
    """
    devices = _get_input_devices(op)
    if (
        len(devices) > 1
        or len(devices) == 0
        or devices[0] is None
        # and not only_constant_inputs
    ):
        raise ValueError(f"Op {op} has input devices {devices}")
    else:
        state.per_rank_fns[devices[0]].ops.append(op)


def _send_projector(op: Op, state: ProjectorState):
    from_d = op.inputs[0].type.device
    to_d = op.attributes["device"]
    assert from_d != to_d
    group = _make_group((from_d, to_d))
    if not isinstance(op.inputs[0].type, Tensor):
        output_shape = tuple()
        output_type = op.inputs[0].type
    else:
        output_shape = op.inputs[0].type.shape
        output_type = op.inputs[0].type.dtype
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
                "shape": output_shape,
                "from_d": from_d,
                "group": group,
                "dtype": output_type,
            },
        )
    )


_ProjectorRegister = {
    ("Add", (Tensor, Tensor)): _identity_projector,
    ("Add", (Tensor, Float32)): _identity_projector,
    ("Cast", (Tensor,)): _identity_projector,
    ("Cast", (Int64,)): _identity_projector,
    ("Cast", (Float64,)): _identity_projector,
    ("Concat", (Tensor, Tensor)): _identity_projector,
    ("Concat", (Tensor, Tensor, Tensor)): _identity_projector,
    ("Concat", (Tensor, Tensor, Tensor, Tensor)): _identity_projector,
    ("Constant", ()): _constant_projector,
    ("ConstantOfShape", (Tensor,)): _identity_projector,
    ("Div", (Tensor, Tensor)): _identity_projector,
    ("Div", (Tensor, Float32)): _identity_projector,
    ("Div", (Int64, Int64)): _identity_projector,
    ("Identity", (Tensor,)): _identity_projector,
    ("Gather", (Tensor, Tensor)): _identity_projector,
    ("Gather", (Tensor, Int64)): _identity_projector,
    ("Gemm", (Tensor, Tensor, Tensor)): _identity_projector,
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
    ("Mul", (Tensor, Tensor)): _identity_projector,
    ("Mul", (Tensor, Float32)): _identity_projector,
    ("Mul", (Int64, Int64)): _identity_projector,
    ("NonZero", (Tensor,)): _identity_projector,
    ("Pow", (Tensor, Float32)): _identity_projector,
    ("ReduceMean", (Tensor,)): _identity_projector,
    ("Relu", (Tensor,)): _identity_projector,
    ("ReluGrad", (Tensor, Tensor)): _identity_projector,
    ("Reshape", (Tensor, Tensor)): _identity_projector,
    ("Shape", (Tensor,)): _identity_projector,
    ("Send", (Tensor,)): _send_projector,
    ("Send", (Int64,)): _send_projector,
    ("Slice", (Tensor, Tensor, Tensor, Tensor, Int64)): _identity_projector,
    ("Softmax", (Tensor,)): _identity_projector,
    ("Split", (Tensor,)): _identity_projector,
    ("Squeeze", (Tensor,)): _identity_projector,
    ("Sqrt", (Tensor,)): _identity_projector,
    ("Sub", (Tensor, Tensor)): _identity_projector,
    ("Sub", (Int64, Int64)): _identity_projector,
    ("Sub", (Float32, Tensor)): _identity_projector,
    ("Tanh", (Tensor,)): _identity_projector,
    ("Transpose", (Tensor,)): _identity_projector,
    ("Unsqueeze", (Tensor,)): _identity_projector,
    ("Unsqueeze", (Int64,)): _identity_projector,
}


# Make semantics of projector functions
_ProjectorSemantics = {}
update_semantics_with_register(_ProjectorSemantics, _ProjectorRegister)


def project(
    fn: Function, input_types: Sequence[Type]
) -> Tuple[Dict[Device, Function], Set[Tuple[Device]]]:
    """Project `fn` to per-rank functions. Returns a mapping from Devices to
    per-rank Functions, and a set of Device groups that perform collective
    communications in `fn`.
    """
    state = ProjectorState(fn, input_types)

    devices = sorted(set(typ.device for typ in input_types))
    for d in devices:
        state.per_rank_fns[d] = FunctionMaker(name=fn.name)

    # Project fn's inputs to each per-rank fn:
    for v, typ in zip(fn.inputs, input_types):
        state.per_rank_fns[typ.device].inputs.append(v)

    # First, interpret the function on inputs to get all values
    state = interpreter.interpret(fn, input_types, state)

    # Then, run each op's projector function
    for op in fn.ops:
        # Find the op's inputs & outputs in state's environment
        inputs = tuple(state.env[v] for v in op.inputs)
        outputs = tuple(state.env[v] for v in op.outputs)

        # Dispatch to find projector function for op
        projector = dispatch(_ProjectorSemantics, op.op_type, inputs)
        # Project op and add to appropriate per-rank function
        projector(op, state)

        # If op involves more than one device, create a group
        devices = [v.device for v in outputs] + [v.type.device for v in op.inputs]
        group = _make_group(devices)
        if len(group) > 1:
            state.groups.add(group)

    result_fns = {}
    for d, per_rank_fn in state.per_rank_fns.items():
        result_fns[d] = per_rank_fn.finalize()

    return result_fns, state.groups
