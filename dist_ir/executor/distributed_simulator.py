from ..ir.type import Tensor


class DistributedSimulator:
    def __init__(self, topology, per_op_costs):
        self._topology = topology
        self._per_op_costs = per_op_costs

    def simulate(self, module):
        peak_memory = {}
        live_memory = {}
        consumers = {}
        timestamps = {}
        devices = self._topology.get_devices()

        for device in devices:
            timestamps[device] = 0.0
            peak_memory[device] = 0.0
            live_memory[device] = 0.0

        for op_name, op in module.get_ops().items():
            in_edges = op.get_in_edges()
            out_edges = op.get_out_edges()

            # Compute the time to transfer the input values.
            for in_edge in in_edges:
                if isinstance(in_edge.type, Tensor):
                    input_data_size = in_edge.type.size() * in_edge.type.dtype.size
                    bandwidth = self._topology.get_bandwidth(in_edge.device, op.device)
                    transfer_time = input_data_size / bandwidth
                    if not module.is_input(in_edge.name):
                        consumers[in_edge.name] -= 1
                    timestamps[in_edge.device] += transfer_time
                    timestamps[op.device] = max(
                        timestamps[in_edge.device], timestamps[op.device]
                    )

            # Free any unused memory.
            for in_edge in in_edges:
                if in_edge.name not in consumers:
                    continue
                input_data_size = in_edge.type.size() * in_edge.type.dtype.size
                if consumers[in_edge.name] == 0:
                    live_memory[in_edge.device] -= input_data_size

            # Compute the time to execute the op.
            # TODO: Incorporate the input shapes into the cost model
            runtime = self._per_op_costs[op.op_type][op.device.device_type]
            timestamps[op.device] += runtime

            # Update the memory usage on each device.
            for out_edge in out_edges:
                if isinstance(out_edge.type, Tensor):
                    output_data_size = out_edge.type.size() * out_edge.type.dtype.size
                    live_memory[out_edge.device] += output_data_size
                    consumers[out_edge.name] = module.get_consumers_for_out_edge(
                        out_edge.name
                    )
            for device in devices:
                if live_memory[device] > peak_memory[device]:
                    peak_memory[device] = live_memory[device]

        execution_time = max([timestamps[device] for device in timestamps])
        print(f"Execution time: {execution_time}")
        print("Peak memory usage per device:")
        for device in sorted(peak_memory.keys()):
            print(f"{device}: {peak_memory[device]}")
