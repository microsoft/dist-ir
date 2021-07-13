from copy import deepcopy
from collections import defaultdict
import json
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from ..ir import Function, Device, Op
from ..ir.type import Type, Tensor
from .absint import AbstractState, AbstractInterpreter
from .numpy_register import NumPyRegister
from .type_inference import TypePropRegister
from .mixed_register import MixedImplementations

SECONDS_TO_MICROSECONDS = 1e6


class SimulatorState(AbstractState):
    def __init__(self, function: Function, inputs: Sequence[Any]):
        AbstractState.__init__(self, function, inputs)
        self.timestamps = defaultdict(float)
        self.peak_memory = defaultdict(lambda: 0)
        # Values are tuples of (device, memory_used)
        self.live_memory = defaultdict(lambda: [(0, 0)])
        self.consumers = defaultdict(int)
        self.trace = []
        self._function_inputs_set = set(function.inputs)

        for inp in inputs:
            self.peak_memory[inp.device] += inp.size()
        for device in self.peak_memory:
            self.live_memory[device][0] = (0, self.peak_memory[device])

    def add_trace_event(self, op_type, device, start_time, duration):
        if device is None:
            raise ValueError(f"No device specified for {op_type} op trace event")

        self.trace.append(
            {
                "name": op_type,
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


def _update_live_memory(state, deltas):
    for device in deltas:
        state.live_memory[device].append(
            (
                state.timestamps[device],
                state.live_memory[device][-1][1] + deltas[device],
            )
        )


def _simulate_op(
    state: SimulatorState,
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

    # Update the trace and timestamps.
    for device in costs:
        state.add_trace_event(
            op.op_type,
            device,
            state.timestamps[device],
            costs[device],
        )
        state.timestamps[device] += costs[device]

    # Update the live memory with any new activations.
    live_memory_deltas = defaultdict(lambda: 0)
    for function_output, output_type in zip(op.outputs, outputs):
        state.consumers[function_output] = len(
            state.function.consumers[function_output]
        )
        output_devices = output_type.get_all_devices()
        for output_device in output_devices:
            live_memory_deltas[output_device] += output_type.size()
    _update_live_memory(state, live_memory_deltas)

    # Update the peak memory.
    for device in state.live_memory:
        state.peak_memory[device] = max(
            state.peak_memory[device], state.live_memory[device][-1][1]
        )

    # Update the live memory to reflect any freed activations.
    live_memory_deltas = defaultdict(lambda: 0)
    for inp, input_type in zip(op.inputs, inputs):
        # We don't free live memory for function inputs as these could be for weights
        # or input data buffers that are active for the entire duration of execution.
        if inp in state._function_inputs_set:
            continue
        if state.consumers[inp] <= 0:
            raise RuntimeError(
                f"Input {in_edge} for op {op} has "
                f"{state.consumers[in_edge]} consumers"
            )
        state.consumers[inp] -= 1
        if state.consumers[inp] == 0:
            input_devices = input_type.get_all_devices()
            for input_device in input_devices:
                live_memory_deltas[input_device] -= input_type.size()
    _update_live_memory(state, live_memory_deltas)


def _create_semantics(cost_functions, implementations):
    """Creates a semantics (dictionary mapping op signatures to abstract state
    modifiers) given a dictionary of cost functions (input values -> costs) and
    a dictionary of implementations (input values -> output values).
    """

    def convert_impl(impl_fn, cost_fn):
        def semantics(op: Op, state: SimulatorState):
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


def Simulator(cost_model):
    return AbstractInterpreter(
        SimulatorState,
        _create_semantics(
            cost_model.cost_functions,
            {**NumPyRegister, **MixedImplementations, **TypePropRegister},
        ),
    )


# TODO: Remove once we have simulation with mixed types
def _create_post_type_inference_semantics(cost_functions):
    """Creates a semantics (dictionary mapping op signatures to abstract state
    modifiers) given a dictionary of cost functions (input values -> costs) and
    a dictionary of implementations (input values -> output values).
    """

    def convert_impl(cost_fn):
        def semantics(op: Op, state: SimulatorState):
            # Find the op's inputs in state's environment
            inputs = tuple(state.env[v] for v in op.inputs)
            outputs = tuple(x.type for x in op.outputs)

            # Run the cost function
            costs = cost_fn(op, *inputs)

            for x in op.outputs:
                state.env[x] = x.type

            _simulate_op(state, op, costs, inputs, outputs)

        return semantics

    signatures = cost_functions.keys()

    return {f: convert_impl(cost_functions[f]) for f in signatures}


def PostTypeInferenceSimulator(cost_model):
    return AbstractInterpreter(
        SimulatorState,
        _create_post_type_inference_semantics(cost_model.cost_functions),
    )


# TODO: Remove once we have simulation with mixed types
def _create_post_type_inference_semantics(cost_functions):
    """Creates a semantics (dictionary mapping op signatures to abstract state
    modifiers) given a dictionary of cost functions (input values -> costs) and
    a dictionary of implementations (input values -> output values).
    """

    def convert_impl(cost_fn):
        def semantics(op: Op, state: SimulatorState):
            # Find the op's inputs in state's environment
            inputs = tuple(state.env[v] for v in op.inputs)
            outputs = tuple(x.type for x in op.outputs)

            # Run the cost function
            costs = cost_fn(op, *inputs)

            for x in op.outputs:
                state.env[x] = x.type

            _simulate_op(state, op, costs, inputs, outputs)

        return semantics

    signatures = cost_functions.keys()

    return {f: convert_impl(cost_functions[f]) for f in signatures}


def PostTypeInferenceSimulator(cost_model):
    return AbstractInterpreter(
        SimulatorState,
        _create_post_type_inference_semantics(cost_model.cost_functions),
    )
