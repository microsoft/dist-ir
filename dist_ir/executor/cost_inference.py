from . import utils


BYTES_IN_GB = 8.0e9


class CostModel:
    """A cost model -- mapping from op type to cost functions. These cost
    functions expect as input the TODO and output a map from devices to runtime.
    (TODO temporary memory)
    """

    def __init__(self, device_throughputs):
        self._device_throughputs = device_throughputs
        self._op_register = {
            "Allreduce": self._infer_costs_for_allreduce,
            "Broadcast": self._infer_costs_for_broadcast_scatter,
            "MatMul": self._infer_costs_for_matmul,
            "Scatter": self._infer_costs_for_broadcast_scatter,
        }

    @property
    def device_type(self):
        return self._device_type

    def _infer_costs_for_allreduce(self, op, in_edges, out_edges, topology):
        costs = {}
        output_devices = utils.get_all_devices(out_edges)
        for device in output_devices:
            # TODO: Compute cost properly
            costs[device] = 0

        return costs

    def _infer_costs_for_matmul(self, op, in_edges, out_edges, topology):
        costs = {}
        device = in_edges[0].type.device
        # TODO: Verify all input and output devices are the same?
        # TODO: Check this cost computation
        a_matrix_shape = in_edges[0].type.shape
        b_matrix_shape = in_edges[1].type.shape
        flops = 2 * a_matrix_shape[1] * a_matrix_shape[0] * b_matrix_shape[1]
        # TODO: Use a better way of computing runtime from FLOPs
        runtime = flops / self._device_throughputs[device.device_type]
        costs[device] = runtime
        return costs

    def _infer_costs_for_broadcast_scatter(self, op, in_edges, out_edges, topology):
        costs = {}
        input_device = in_edges[0].type.device
        costs[input_device] = 0
        input_size = in_edges[0].type.size() * in_edges[0].type.dtype.size
        input_size_gb = input_size / BYTES_IN_GB
        output_devices = utils.get_all_devices(out_edges)
        for output_device in output_devices:
            bandwidth = topology.get_bandwidth(input_device, output_device)
            transfer_time = input_size_gb / bandwidth
            # NOTE: This assumes all tensors can be sent concurrently
            # TODO: Do we need to model the link capacity?
            costs[input_device] = max(costs[input_device], transfer_time)
            costs[output_device] = transfer_time

        return costs

    def infer_costs(self, op, topology):
        in_edges = op.get_in_edges()
        out_edges = op.get_out_edges()

        return self._op_register[op.op_type](op, in_edges, out_edges, topology)
