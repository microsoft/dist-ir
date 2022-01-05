# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from copy import deepcopy
from collections import defaultdict
import json
import logging
from typing import Any, Dict, Sequence, Set, Tuple

from ..ir import Function, Device, Op
from ..ir.type import Tensor, Type, abstract_values
from .absint import (
    AbstractState,
    interpreter,
    update_semantics_with_register,
    dispatch,
)
from .concrete_value import ConcreteValue
from .cost_model import CostModel, KERNEL_LAUNCH_OVERHEAD

SECONDS_TO_MICROSECONDS = 1e6


def _get_all_devices(values: Sequence[Any]) -> Set[Device]:
    """Returns the devices that `values` live on. `values` can be any valid
    abstract interpreter values, e.g., any instance of Type or ConcreteValue."""
    devices = set()
    for v in values:
        if isinstance(v, Type):
            devices.update(v.get_all_devices())
        elif isinstance(v, ConcreteValue):
            devices.add(v.device)
        else:
            raise ValueError(f"_get_all_devices called on value {v} of type {type(v)}")
    return devices


class SimulatorState(AbstractState):
    # TODO remove subclass, unnecessary init args?
    def __init__(self, function: Function, inputs: Sequence[Any]):
        AbstractState.__init__(self, function, inputs)
        self.timestamps = defaultdict(float)
        self.peak_memory = defaultdict(lambda: 0)
        # Values are tuples of (device, memory_used)
        self.live_memory = defaultdict(lambda: [(0, 0)])
        self.consumers = defaultdict(int)
        self.trace = []
        self._function_inputs_set = set(function.inputs)

        for i, inp in enumerate(function.inputs):
            if inp.type is None or inp.type.device is None:
                if (
                    isinstance(inputs[i], ConcreteValue)
                    and inputs[i].device is not None
                ):
                    cur_live_memory = self.live_memory[inputs[i].device][0][-1]
                    cur_live_memory += inputs[i].val.nbytes
                    self.live_memory[inputs[i].device][0] = (0, cur_live_memory)
                elif (
                    isinstance(inputs[i], Tensor)
                    and inputs[i].shape is not None
                    and inputs[i].dtype is not None
                    and inputs[i].device is not None
                ):
                    cur_live_memory = self.live_memory[inputs[i].device][0][-1]
                    cur_live_memory += inputs[i].size()
                    self.live_memory[inputs[i].device][0] = (0, cur_live_memory)
                else:
                    logging.warning(
                        f"No input type or device for input {inp} ({type(inputs[i])})"
                    )
                    continue
            else:
                cur_live_memory = self.live_memory[inp.type.device][0][-1]
                cur_live_memory += inp.type.size()
                self.live_memory[inp.type.device][0] = (0, cur_live_memory)
        for device in self.peak_memory:
            self.live_memory[device][0] = (0, self.peak_memory[device])

    def get_latency(self):
        return max([self.timestamps[d] for d in self.timestamps])

    def get_throughput(self, batch_size):
        return batch_size / self.get_latency()

    def set_peak_memory(self):
        for device in self.live_memory:
            self.peak_memory[device] = max(
                [
                    self.live_memory[device][i][1]
                    for i in range(len(self.live_memory[device]))
                ]
            )

    def get_peak_memory(self):
        return max([self.peak_memory[d] for d in self.peak_memory])

    def print_summary(self, batch_size):
        logging.info(f"Latency: {self.get_latency()} seconds")
        logging.info(
            f"Throughput: {self.get_throughput(batch_size):.2f} samples / second"
        )
        logging.info(f"Peak memory: {self.get_peak_memory() / 1e9:.2f} GB")

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

    def update_live_memory(self, deltas):
        for device in deltas:
            self.live_memory[device].append(
                (
                    self.timestamps[device],
                    self.live_memory[device][-1][1] + deltas[device],
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
    devices = _get_all_devices(inputs + outputs)
    if len(devices) > 1:
        max_timestamp = max([state.timestamps[device] for device in devices])
        for device in devices:
            state.timestamps[device] = max_timestamp

    # Update the trace and timestamps
    for device in costs:
        state.add_trace_event(
            op.op_type,
            device,
            state.timestamps[device],
            costs[device],
        )
        state.timestamps[device] += costs[device]

    # Update the live memory with any new activations.
    # We skip updating memory for SGDOptimizer outputs because we
    # assume in-place weight updates.
    # TODO: Handle in-place ops more generally?
    if op.op_type != "SGDOptimizer":
        live_memory_deltas = defaultdict(lambda: 0)
        for output, out_edge in zip(outputs, op.outputs):
            state.consumers[out_edge] = len(state.function.consumers[out_edge])
            output_devices = _get_all_devices([output])
            for output_device in output_devices:
                live_memory_deltas[output_device] += output.size()
        state.update_live_memory(live_memory_deltas)

    # Update the live memory to reflect any freed activations.
    live_memory_deltas = defaultdict(lambda: 0)
    for inp, in_edge in zip(inputs, op.inputs):
        # We don't free live memory for function inputs as these could be for weights
        # or input data buffers that are active for the entire duration of execution.
        if in_edge in state._function_inputs_set:
            continue
        if state.consumers[in_edge] <= 0:
            raise RuntimeError(
                f"Input {in_edge} for op {op} has "
                f"{state.consumers[in_edge]} consumers"
            )
        state.consumers[in_edge] -= 1
        if state.consumers[in_edge] == 0:
            input_devices = _get_all_devices([inp])
            for input_device in input_devices:
                live_memory_deltas[input_device] -= inp.size()
    state.update_live_memory(live_memory_deltas)


class Simulator:
    def __init__(
        self,
        cost_model: CostModel,
        # self, cost_functions: Dict[Tuple[str, Tuple[type, ...]], Callable]
    ) -> None:
        # Make semantics of cost_functions
        self.cost_functions = {}
        update_semantics_with_register(self.cost_functions, cost_model.cost_functions)

    def simulate(self, function: Function, inputs: Tuple[Any]) -> SimulatorState:
        """Simulate `function` on `inputs`.

        `inputs` is a tuple of abstract interpreter values (abstract or concrete).

        Returns a SimulatorState containing timestamps, memory profiles, etc.
        """
        state = SimulatorState(function, inputs)

        # First, interpret the function on inputs to get all values
        state = interpreter.interpret(function, inputs, state)

        # Then, run each op's cost function
        for op in function.ops:
            # Find the op's inputs & outputs in state's environment
            inputs = tuple(state.env[v] for v in op.inputs)
            outputs = tuple(state.env[v] for v in op.outputs)

            # Dispatch to find cost function for op
            signature = None
            try:
                signature, cost_function = dispatch(
                    self.cost_functions, op.op_type, inputs
                )
            except ValueError:
                warn(f"Dispatch failed for op {op.op_type} on inputs {inputs}")
            if signature is None:
                # Use default cost function if signature not in cost_functions
                devices = _get_all_devices(inputs + outputs)
                costs = {device: KERNEL_LAUNCH_OVERHEAD for device in devices}
            else:
                # Abstract inputs if necessary
                abstracted_inputs = abstract_values(inputs, signature)
                costs = cost_function(op, *abstracted_inputs)

            _simulate_op(state, op, costs, inputs, outputs)

        # Record the peak memory.
        state.set_peak_memory()

        return state
