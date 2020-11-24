from ..ir.type import Tensor
from . import utils
from .shape_inference import ShapeInferenceRegister

import copy


class DistributedSimulator:
    def __init__(self, topology, per_op_costs):
        self._topology = topology
        self._per_op_costs = per_op_costs

    def _simulate_pmap(
        self, op, inputs, outputs, timestamps, consumers, peak_memory, live_memory
    ):
        value_name_map = op.get_metadata("value_name_map")
        value_map = {}
        for input in inputs:
            value_map[input.name] = input
        for output in outputs:
            value_map[output.name] = output
        submodule = op.get_submodule(0)
        for device in value_name_map:
            per_device_submodule = copy.deepcopy(submodule)
            for op_name, op in per_device_submodule.get_ops().items():
                op.device = device
            self._simulate(
                per_device_submodule,
                timestamps,
                consumers,
                peak_memory,
                live_memory,
                value_name_map,
                value_map,
            )

    def _simulate(
        self,
        module,
        timestamps,
        consumers,
        peak_memory,
        live_memory,
        value_name_map=None,
        value_map=None,
    ):
        devices = self._topology.get_devices()

        for op_name, op in module.get_ops().items():
            inputs = op.get_in_edges()
            outputs = op.get_out_edges()

            mapped_inputs = utils.map_values(
                inputs, value_name_map, value_map, op.device
            )
            mapped_outputs = utils.map_values(
                outputs, value_name_map, value_map, op.device
            )

            if op.op_type == "Pmap":
                self._simulate_pmap(
                    op,
                    mapped_inputs,
                    mapped_outputs,
                    timestamps,
                    consumers,
                    peak_memory,
                    live_memory,
                )
            else:
                ShapeInferenceRegister[op.op_type](op, mapped_inputs, mapped_outputs)

                # Compute the time to transfer the input values.
                for in_edge in mapped_inputs:
                    if isinstance(in_edge.type, Tensor):
                        input_data_size = in_edge.type.size() * in_edge.type.dtype.size
                        if in_edge.device != op.device:
                            live_memory[op.device] += input_data_size
                            if live_memory[op.device] > peak_memory[op.device]:
                                peak_memory[op.device] = live_memory[device]
                        bandwidth = self._topology.get_bandwidth(
                            in_edge.device, op.device
                        )
                        transfer_time = input_data_size / bandwidth
                        if not module.is_input(in_edge.name):
                            consumers[in_edge.name] -= 1
                        timestamps[in_edge.device] += transfer_time
                        timestamps[op.device] = max(
                            timestamps[in_edge.device], timestamps[op.device]
                        )

                # Free any unused memory.
                for in_edge in mapped_inputs:
                    if in_edge.name not in consumers:
                        continue
                    input_data_size = in_edge.type.size() * in_edge.type.dtype.size
                    if consumers[in_edge.name] == 0:
                        live_memory[in_edge.device] -= input_data_size

                # Compute or look up the time to execute the op.
                runtime = self._per_op_costs[op.op_type][op.device.device_type]
                timestamps[op.device] += runtime

            # Update the memory usage on each device.
            for out_edge in mapped_outputs:
                if isinstance(out_edge.type, Tensor):
                    output_data_size = out_edge.type.size() * out_edge.type.dtype.size
                    live_memory[out_edge.device] += output_data_size
                    consumers[out_edge.name] = module.get_consumers_for_out_edge(
                        out_edge.name
                    )
            for device in devices:
                if live_memory[device] > peak_memory[device]:
                    peak_memory[device] = live_memory[device]

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
            if isinstance(input.type, Tensor):
                live_memory[input.device] += input.type.size() * input.type.dtype.size

        for device in live_memory:
            peak_memory[device] = live_memory[device]

        return self._simulate(module, timestamps, consumers, peak_memory, live_memory)
