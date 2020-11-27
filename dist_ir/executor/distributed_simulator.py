from ..ir.type import Tensor
from . import utils
from .shape_inference import ShapeInferenceRegister

import copy


class DistributedSimulatorState:
    def __init__(self):
        self.timestamps = {}
        self.consumers = {}
        self.peak_memory = {}
        self.live_memory = {}

    def update(self, other):
        def update_state_dict(a, b):
            for device in b:
                for bound_device in device.bound_devices:
                    a[bound_device] += b[device]

        update_state_dict(self.timestamps, other.timestamps)
        update_state_dict(self.peak_memory, other.peak_memory)
        update_state_dict(self.live_memory, other.live_memory)

        # TODO: Update consumers?


class DistributedSimulator:
    def __init__(self, topology, cost_model):
        self._topology = topology
        self._cost_model = cost_model

    def _simulate(self, module, state):

        for op_name, op in module.get_ops().items():
            in_edges = op.get_in_edges()
            out_edges = op.get_out_edges()

            # Synchronize all input and output devices for this op if necessary.
            input_devices = utils.get_all_devices(in_edges)
            output_devices = utils.get_all_devices(out_edges)
            input_and_output_devices = input_devices.union(output_devices)
            if len(input_and_output_devices) > 1:
                max_timestamp = max(
                    [
                        0
                        if device not in state.timestamps
                        else state.timestamps[device]
                        for device in input_and_output_devices
                    ]
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
                state.update(submodule_state)
            else:
                costs = self._cost_model.infer_costs(op, self._topology)
                for device in costs:
                    if device not in state.timestamps:
                        state.timestamps[device] = 0
                    state.timestamps[device] += costs[device]

            # Update the live memory.
            for out_edge in out_edges:
                state.consumers[out_edge] = module.get_consumers_for_out_edge(
                    out_edge.name
                )
                output_devices = out_edge.type.get_all_devices()
                for output_device in output_devices:
                    if output_device not in state.live_memory:
                        state.live_memory[output_device] = 0
                    state.live_memory[output_device] += out_edge.type.size()
            values_to_free = []
            for value in state.consumers:
                if state.consumers[value] == 0 and not module.is_input(value.name):
                    devices = value.type.get_all_devices()
                    for device in devices:
                        state.live_memory[device] -= value.type.size()
                    values_to_free.append(value)
            for value_to_free in values_to_free:
                del state.consumers[value_to_free]

            # Update the peak memory.
            for device in state.live_memory:
                state.peak_memory[device] = max(
                    0 if device not in state.peak_memory else state.peak_memory[device],
                    state.live_memory[device],
                )

    def simulate(self, module):
        state = DistributedSimulatorState()
        self._simulate(module, state)
        return state
