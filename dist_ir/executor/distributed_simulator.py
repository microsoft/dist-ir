from copy import deepcopy
from collections import defaultdict
from dist_ir.executor.type_inference import TypePropRegister
import json
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from ..ir import Function, Device, Op
from ..ir.type import Type, Tensor
from . import utils
from .absint import AbstractState, AbstractInterpreter
from .numpy_register import NumPyRegister

SECONDS_TO_MICROSECONDS = 1e6

# TODO rename: distributed simulator -> simulator


class DistributedSimulatorState(AbstractState):
    def __init__(self, function: Function, inputs: Sequence[Any]):
        AbstractState.__init__(self, function, inputs)
        self.timestamps = defaultdict(float)
        self.peak_memory = defaultdict(float)
        self.live_memory = defaultdict(float)
        self.consumers = defaultdict(int)
        self.trace = []

    def add_trace_event(self, op_name, device, start_time, duration):
        self.trace.append(
            {
                "name": op_name,
                "ph": "X",
                "ts": start_time,
                "dur": duration,
                "pid": 0,
                "tid": device.device_id,
            }
        )

    def dump_chrome_trace(self, fname):
        # Chrome trace expects times in microseconds
        _trace = deepcopy(self.trace)
        for event in _trace:
            event["ts"] = event["ts"] * SECONDS_TO_MICROSECONDS
            event["dur"] = event["dur"] * SECONDS_TO_MICROSECONDS

        with open(fname, "w") as fout:
            json.dump(_trace, fout, indent=0)


def _simulate_op(
    state: DistributedSimulatorState,
    op: Op,
    costs: Dict[Device, float],
    inputs: Tuple[Any],
    outputs: Tuple[Any],
):
    # Synchronize all input and output devices for this op.
    # TODO need to do something more robust here, because some input/output
    # values are np.ndarrays, which don't have device fields.
    # For e.g., we could wrap all abstract values in some AbstractValue class,
    # and attach the device tag to this class.
    devices = set()
    for v in inputs + outputs:
        if isinstance(v, Type):
            devices.update(v.get_all_devices())
    if len(devices) > 1:
        max_timestamp = max([state.timestamps[device] for device in devices])
        for device in devices:
            state.timestamps[device] = max_timestamp

    # Update the trace and timestamps
    for device in costs:
        state.add_trace_event(
            op.name,
            device,
            state.timestamps[device],
            costs[device],
        )
        state.timestamps[device] += costs[device]

    # Update the live memory.
    for out_edge in op.outputs:
        state.consumers[out_edge] = len(state.function.consumers[out_edge])
        # Output value could live on multiple devices (e.g. scatter) so
        # update memory on all devices:
        output_devices = out_edge.type.get_all_devices()
        for output_device in output_devices:
            state.live_memory[output_device] += out_edge.type.size()
    # TODO: Can we optimize this using a priority queue?
    for value in state.consumers:
        # TODO we are missing a decrement of state.consumers[value] somewhere
        if state.consumers[value] == 0 and all(
            value != v for v in state.function.inputs
        ):
            value_devices = value.type.get_all_devices()
            for device in value_devices:
                state.live_memory[device] -= value.type.size()

    # Update the peak memory.
    for device in state.live_memory:
        state.peak_memory[device] = max(
            state.peak_memory[device], state.live_memory[device]
        )


def _create_semantics(cost_functions, implementations):
    """Creates a semantics (dictionary mapping op signatures to abstract state
    modifiers) given a dictionary of cost functions (input values -> costs) and
    a dictionary of implementations (input values -> output values).
    """

    def convert_impl(impl_fn, cost_fn):
        def semantics(op: Op, state: DistributedSimulatorState):
            # Find the op's inputs in state's environment
            inputs = tuple(state.env[v] for v in op.inputs)

            # Run the abstract/concrete implementation
            outputs = impl_fn(op, *inputs)

            # Run the cost function
            costs = cost_fn(op, *inputs)

            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for x, val in zip(op.outputs, outputs):
                state.env[x] = val

            _simulate_op(state, op, costs, inputs, outputs)

        return semantics

    signatures = set(cost_functions.keys()).intersection(implementations.keys())

    return {f: convert_impl(implementations[f], cost_functions[f]) for f in signatures}


# All these cost functions assume they are getting the type of each input value
# TODO instead of passing the op, should we pass the attributes as kwargs?


# Some "mixed" abstract/concrete implementations of ops that are needed for
# more precise simulation:
# TODO what's the right place for these?


def _shape_abstract_to_concrete(op, x: Tensor):
    return np.array(x.shape, dtype=np.int64)


def _matmul_abstract(op, x, y):
    if not (x.dtype == y.dtype and x.device == y.device and x.shape[1] == y.shape[0]):
        raise Exception
        # _raise_type_error(op, x, y)
    return Tensor(dtype=x.dtype, shape=(x.shape[0], y.shape[1]), device=x.device)


def _slice_abstract_exact(op, x, starts, ends, axes):
    """The case when we know the slice indices concretely but x is abstract."""
    # TODO handle the other cases, e.g. negative indices
    slices = {axis: slice(s, e) for (s, e, axis) in zip(starts, ends, axes)}
    slices = tuple(slices.get(d, slice(None)) for d in range(len(x.shape)))
    # Create a fake tensor and slice it because I'm lazy to work out the new shape
    y = np.zeros(x.shape)
    return Tensor(dtype=x.dtype, shape=y[slices].shape, device=x.device)


MixedImplementations = {
    ("MatMul", (Tensor, Tensor)): _matmul_abstract,
    ("Shape", (Tensor,)): _shape_abstract_to_concrete,
    ("Slice", (Tensor, np.ndarray, np.ndarray, np.ndarray)): _slice_abstract_exact,
}


def Simulator(cost_model):
    return AbstractInterpreter(
        DistributedSimulatorState,
        _create_semantics(
            cost_model.cost_functions,
            {**NumPyRegister, **MixedImplementations, **TypePropRegister},
        ),
    )
