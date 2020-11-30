from ..ir import Module
from ..ir.type import Tensor
from . import utils
from .shape_inference import ShapeInferenceRegister

import copy
from collections import defaultdict


class DistributedSimulatorState:
    def __init__(self):
        self.timestamps = defaultdict(lambda: 0.0)
        self.consumers = defaultdict(lambda: 0)
        self.peak_memory = defaultdict(lambda: 0.0)
        self.live_memory = defaultdict(lambda: 0.0)


class DistributedSimulator:
    def __init__(self, cost_model):
        self._cost_model = cost_model

    def _simulate(self, module: Module, state: DistributedSimulatorState):

        for op_name, op in module.get_ops().items():
            in_edges = op.get_in_edges()
            out_edges = op.get_out_edges()

            # Synchronize all input and output devices for this op if necessary.
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
                bound_devices = op.get_attribute("devices")
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
