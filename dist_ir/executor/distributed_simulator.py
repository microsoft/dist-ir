from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
import json
from typing import Any, Dict, Sequence

from ..ir import Function, Device, Op
from . import utils
from .absint import (
    AbstractState,
    AbstractInterpreter,
    MixedImplementations,
)

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


def _simulate_op(state: DistributedSimulatorState, op: Op, costs: Dict[Device, float]):
    # Synchronize all input and output devices for this op.
    input_devices = utils.get_all_devices(op.inputs)
    output_devices = utils.get_all_devices(op.outputs)
    input_and_output_devices = input_devices.union(output_devices)
    if len(input_and_output_devices) > 1:
        max_timestamp = max(
            [state.timestamps[device] for device in input_and_output_devices]
        )
        for device in input_and_output_devices:
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
        # TODO are we missing a decrement of state.consumers[value] somewhere?
        if state.consumers[value] == 0 and all(
            value != v for v in state.function.inputs
        ):
            devices = value.type.get_all_devices()
            for device in devices:
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

            _simulate_op(state, op, costs)

        return semantics

    signatures = set(cost_functions.keys()).intersection(implementations.keys())

    return {f: convert_impl(implementations[f], cost_functions[f]) for f in signatures}


# All these cost functions assume they are getting the type of each input value
# TODO instead of passing the op, should we pass the attributes as kwargs?


def Simulator(cost_model):
    return AbstractInterpreter(
        DistributedSimulatorState,
        _create_semantics(cost_model.cost_functions, MixedImplementations),
    )
