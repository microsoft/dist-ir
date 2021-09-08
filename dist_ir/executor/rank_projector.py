# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from typing import Any, Dict, Sequence, Set, Tuple

from ..ir import Function, FunctionMaker, Device, Op
from ..ir.type import Type, Float32, Float64, Int64, Tensor, abstract_values
from .absint import (
    AbstractState,
    dispatch,
    interpreter,
    update_semantics_with_register,
)
from .concrete_value import ConcreteValue

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


def _get_devices(inputs: Tuple[Any]):
    return list(set(x.device for x in inputs if x.device is not None))


def _make_group(devices):
    """Return a hashable representation of a group of devices. This is needed by
    the backend, which maps them to process groups for communication primitives.
    """
    return tuple(sorted(set(devices)))


# Projector functions:
# Each function takes (op, state, inputs, outputs) and adds the projected version
# of the op to the appropriate per-rank functions. The inputs and outputs arguments
# are the mixed (abstract/concrete) values from the interpreter run used to
# figure out device placements and required shapes (e.g. _send_projector
# needs to know the shape of the sent tensor).


def _collective_projector(op: Op, state: ProjectorState, inputs, outputs):
    """Projects a collective op over D devices that has D inputs and D outputs,
    one on each device."""
    assert len(inputs) == len(outputs)
    for in_v, out_v in zip(inputs, outputs):
        assert in_v.device == out_v.device
    group = _make_group(v.device for v in inputs + outputs)
    attributes = {
        **(op.attributes if op.attributes is not None else {}),
        "group": group,
    }
    for in_, in_v, out_v in zip(inputs, op.inputs, op.outputs):
        d = in_.device

        new_op = Op(
            op.op_type,
            inputs=(in_v,),
            output_values=(out_v,),
            attributes=attributes,
        )
        state.per_rank_fns[d].ops.append(new_op)


def _constant_projector(op: Op, state: ProjectorState, inputs, outputs):
    assert len(op.outputs) == 1
    device = op.attributes["device"]
    state.per_rank_fns[device].ops.append(op)


def _gather_projector(op: Op, state: ProjectorState, inputs, outputs):
    devices = set(v.device for v in inputs)
    assert len(op.inputs) == len(devices)
    assert len(op.outputs) == 1 and outputs[0].device in devices
    attributes = {
        **(op.attributes if op.attributes is not None else {}),
        "group": _make_group(devices),
    }
    for in_, in_v in zip(inputs, op.inputs):
        d = in_.device
        new_op = Op(
            op.op_type,
            inputs=(in_v,),
            output_values=op.outputs,  # TODO only on dst device!
            attributes=attributes,
        )
        state.per_rank_fns[d].ops.append(new_op)


def _identity_projector(op: Op, state: ProjectorState, inputs, outputs):
    """Projects op unchanged to its device's per-rank program.
    The inputs of op must all be on a single device.
    """
    devices = _get_devices(inputs)
    if (
        len(devices) > 1
        or len(devices) == 0
        or devices[0] is None
        # and not only_constant_inputs
    ):
        raise ValueError(f"Op {op} has input devices {devices}")
    else:
        state.per_rank_fns[devices[0]].ops.append(op)


def _send_projector(op: Op, state: ProjectorState, inputs, outputs):
    inp = inputs[0]
    from_d = inp.device
    to_d = op.attributes["device"]
    assert from_d != to_d
    group = _make_group((from_d, to_d))
    if not isinstance(inp, Tensor) and not isinstance(inp, ConcreteValue):
        # Input could be a primitive type
        output_shape = tuple()
        output_type = inp
    else:
        if isinstance(inp, ConcreteValue):
            inp = abstract_values((inp,), (Tensor,))[0]
        output_shape = inp.shape
        output_type = inp.dtype
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
    "Add": _identity_projector,
    "Cast": _identity_projector,
    "Concat": _identity_projector,
    "Constant": _constant_projector,
    "ConstantOfShape": _identity_projector,
    "Div": _identity_projector,
    "Identity": _identity_projector,
    "Gather": _identity_projector,
    "Gemm": _identity_projector,
    "Loss": _identity_projector,
    "LossGrad": _identity_projector,
    "MatMul": _identity_projector,
    "MatMulGrad": _identity_projector,
    "MPIAllgather": _collective_projector,
    "MPIAllreduce": _collective_projector,
    "MPIGather": _gather_projector,
    "Mul": _identity_projector,
    "NonZero": _identity_projector,
    "Pow": _identity_projector,
    "ReduceMean": _identity_projector,
    "Relu": _identity_projector,
    "ReluGrad": _identity_projector,
    "Reshape": _identity_projector,
    "Shape": _identity_projector,
    "Send": _send_projector,
    "Slice": _identity_projector,
    "Softmax": _identity_projector,
    "Split": _identity_projector,
    "Squeeze": _identity_projector,
    "Sqrt": _identity_projector,
    "Sub": _identity_projector,
    "Tanh": _identity_projector,
    "Transpose": _identity_projector,
    "Unsqueeze": _identity_projector,
}


def project(
    fn: Function, input_types: Sequence[Type]
) -> Tuple[Dict[Device, Function], Set[Tuple[Device]]]:
    """Project `fn` to per-rank functions. Uses `input_types` (abstract
    interpreter values, can be abstract or concrete) to infer the devices each
    op executes on.

    Returns a mapping from Devices to per-rank Functions, and a set of Device
    groups that perform collective communications in `fn`.
    """
    state = ProjectorState(fn, input_types)

    devices = sorted(set(typ.device for typ in input_types))
    for d in devices:
        state.per_rank_fns[d] = FunctionMaker(name=fn.name)

    # Project fn's inputs to each per-rank fn:
    for v, typ in zip(fn.inputs, input_types):
        state.per_rank_fns[typ.device].inputs.append(v)

    # First, interpret the function on input_types to get device/shape info
    state = interpreter.interpret(fn, input_types, state)

    # Then, run each op's projector function
    for op in fn.ops:
        # Find the op's inputs & outputs in state's environment
        inputs = tuple(state.env[v] for v in op.inputs)
        outputs = tuple(state.env[v] for v in op.outputs)

        # Dispatch to find projector function for op
        projector = _ProjectorRegister[op.op_type]
        # Project op and add to appropriate per-rank function
        projector(op, state, inputs, outputs)

        # If op involves more than one device, create a group
        devices = [v.device for v in outputs] + [v.device for v in inputs]
        group = _make_group(devices)
        if len(group) > 1:
            state.groups.add(group)

    result_fns = {}
    for d, per_rank_fn in state.per_rank_fns.items():
        result_fns[d] = per_rank_fn.finalize()

    return result_fns, state.groups
