from ..ir.type import Tensor
from . import utils
from .shape_inference import ShapeInferenceRegister

import copy


class DistributedSimulator:
    def __init__(self, topology, cost_model):
        self._topology = topology
        self._cost_model = cost_model

    def _simulate(
        self,
        module,
        timestamps,
        consumers,
        peak_memory,
        live_memory,
    ):

        for op_name, op in module.get_ops().items():
            in_edges = op.get_in_edges()
            out_edges = op.get_out_edges()

            # Synchronize all input devices for this op if necessary.
            input_devices = utils.get_all_devices(in_edges)
            if len(input_devices) > 1:
                max_timestamp = max(
                    [timestamps[input_device] for input_device in input_devices]
                )
                for input_device in input_devices:
                    timestamps[input_device] = max_timestamp

            if op.op_type == "Pmap":
                submodule = op.get_submodule(0)
                self._simulate(
                    submodule, timestamps, consumers, peak_memory, live_memory
                )
            else:
                costs = self._cost_model.infer_costs(op, self._topology)
                for device in costs:
                    timestamps[device] += costs[device]

            for out_edge in out_edges:
                consumers[out_edge] = module.get_consumers_for_out_edge(out_edge.name)
                output_devices = out_edge.type.get_all_devices()
                for output_device in output_devices:
                    live_memory[output_device] += out_edge.type.size()

            values_to_free = []
            for value in consumers:
                if consumers[value] == 0 and not module.is_input(value.name):
                    devices = value.type.get_all_devices()
                    for device in devices:
                        live_memory[device] -= value.type.size()
                    values_to_free.append(value)
            for value_to_free in values_to_free:
                del consumers[value_to_free]

            for device in live_memory:
                peak_memory[device] = max(peak_memory[device], live_memory[device])

        return (timestamps, peak_memory)

    def simulate(self, module):
        timestamps = {}
        consumers = {}
        peak_memory = {}
        live_memory = {}
        devices = self._topology.get_devices()

        for device in devices:
            timestamps[device] = 0.0
            live_memory[device] = 0.0

        for input in module.get_inputs():
            input_devices = input.type.get_all_devices()
            for input_device in input_devices:
                live_memory[input_device] += input.type.size()

        for device in live_memory:
            peak_memory[device] = live_memory[device]

        return self._simulate(module, timestamps, consumers, peak_memory, live_memory)
