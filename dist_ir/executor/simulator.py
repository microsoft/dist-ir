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

        for inp in function.inputs:
            self.peak_memory[inp.type.device] += inp.type.size()
        for device in self.peak_memory:
            self.live_memory[device][0] = (0, self.peak_memory[device])

    def add_trace_event(self, op_type, device, start_time, duration):
        if device is None:
            return
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
    new_live_memory = defaultdict(lambda: 0)
    for out_edge in op.outputs:
        state.consumers[out_edge] = len(state.function.consumers[out_edge])
        output_devices = out_edge.type.get_all_devices()
        for output_device in output_devices:
            new_live_memory[output_device] += out_edge.type.size()
    for device in new_live_memory:
        state.live_memory[device].append(
            (
                state.timestamps[device],
                state.live_memory[device][-1][1] + new_live_memory[device],
            )
        )

    # Update the peak memory.
    for device in state.live_memory:
        state.peak_memory[device] = max(
            state.peak_memory[device], state.live_memory[device][-1][1]
        )

    # Update the live memory to reflect any freed activations.
    freed_live_memory = defaultdict(lambda: 0)
    for in_edge in op.inputs:
        # We don't free live memory for function inputs as these could be for weights
        # or input data buffers that are active for the entire duration of execution.
        if in_edge in state._function_inputs_set:
            continue
        if state.consumers[in_edge] <= 0:
            raise RuntimeError(
                f"Input {in_edge} for op {op} has "
                f"{state.consumers[in_edge]} consumers"
            )
        assert state.consumers[in_edge] > 0
        state.consumers[in_edge] -= 1
        if state.consumers[in_edge] == 0:
            input_devices = in_edge.type.get_all_devices()
            for input_device in input_devices:
                freed_live_memory[input_device] += in_edge.type.size()
    for device in freed_live_memory:
        state.live_memory[device].append(
            (
                state.timestamps[device],
                state.live_memory[device][-1][1] - freed_live_memory[device],
            )
        )


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
        SimulatorState,
        _create_semantics(
            cost_model.cost_functions,
            {**NumPyRegister, **MixedImplementations, **TypePropRegister},
        ),
    )


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
