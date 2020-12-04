from copy import deepcopy
from collections import defaultdict
import json

from ..ir import Module
from . import utils


class DistributedSimulatorState:
    def __init__(self):
        self.timestamps = defaultdict(lambda: 0.0)
        self.consumers = defaultdict(lambda: 0)
        self.peak_memory = defaultdict(lambda: 0.0)
        self.live_memory = defaultdict(lambda: 0.0)
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
        # TODO multiplying by a larger num here to make debugging easier
        _trace = deepcopy(self.trace)
        for event in _trace:
            event["ts"] = int(event["ts"] * 1e11)
            event["dur"] = int(event["dur"] * 1e11)
            # event["ts"] = event["ts"] * 1e6
            # event["dur"] = event["dur"] * 1e6

        with open(fname, "w") as fout:
            json.dump(_trace, fout, indent=0)


class DistributedSimulator:
    def __init__(self, cost_model):
        self._cost_model = cost_model

    def _simulate(self, module: Module, state: DistributedSimulatorState):

        for op_name, op in module.get_ops().items():
            in_edges = op.get_in_edges()
            out_edges = op.get_out_edges()

            # Synchronize all input and output devices for this op.
            input_devices = utils.get_all_devices(in_edges)
            output_devices = utils.get_all_devices(out_edges)
            input_and_output_devices = input_devices.union(output_devices)
            if len(input_and_output_devices) > 1:
                max_timestamp = max(
                    [state.timestamps[device] for device in input_and_output_devices]
                )
                for device in input_and_output_devices:
                    state.timestamps[device] = max_timestamp

            # Compute the costs for the op.
            if op.op_type == "Pmap":
                # For Pmap ops we use a fresh state object and update the enclosing
                # module state using the Pmap state.
                submodule = op.get_submodule(0)
                submodule_state = DistributedSimulatorState()
                self._simulate(submodule, submodule_state)
                device_vars = submodule_state.timestamps.keys()
                assert len(device_vars) == 1
                # TODO what happens when pmaps are nested?
                bound_devices = op.get_attribute("devices")
                # Add submodule's trace to trace of all participating devices
                for device in bound_devices:
                    for event in submodule_state.trace:
                        # Need to add pmap's starting timestamp to event
                        # since submodule_state started at time 0
                        start_time = event["ts"] + state.timestamps[device]
                        state.add_trace_event(
                            event["name"], device, start_time, event["dur"]
                        )
                for device_var in device_vars:
                    for bound_device in bound_devices:
                        state.timestamps[bound_device] += submodule_state.timestamps[
                            device_var
                        ]
                        state.live_memory[bound_device] += submodule_state.live_memory[
                            device_var
                        ]
                        state.peak_memory[bound_device] += submodule_state.peak_memory[
                            device_var
                        ]
                        # TODO: Update consumers?
            else:
                costs = self._cost_model.infer_costs(op)
                for device in costs:
                    state.add_trace_event(
                        op_name,
                        device,
                        state.timestamps[device],
                        costs[device],
                    )
                    state.timestamps[device] += costs[device]

            # Update the live memory.
            for out_edge in out_edges:
                state.consumers[out_edge] = module.get_consumers_for_out_edge(
                    out_edge.name
                )
                # Output value could live on multiple devices (e.g. scatter) so
                # update memory on all devices:
                output_devices = out_edge.type.get_all_devices()
                for output_device in output_devices:
                    state.live_memory[output_device] += out_edge.type.size()
            # TODO: Can we optimize this using a priority queue?
            for value in state.consumers:
                if state.consumers[value] == 0 and not module.is_input(value.name):
                    devices = value.type.get_all_devices()
                    for device in devices:
                        state.live_memory[device] -= value.type.size()

            # Update the peak memory.
            for device in state.live_memory:
                state.peak_memory[device] = max(
                    state.peak_memory[device], state.live_memory[device]
                )

    def simulate(self, module):
        state = DistributedSimulatorState()
        self._simulate(module, state)
        return state
