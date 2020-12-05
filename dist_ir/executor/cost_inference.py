from . import utils

import numpy as np

BYTES_IN_GB = 8.0e9


class CostModel:
    """A cost model -- mapping from op type to cost functions. These cost
    functions expect as input the TODO and output a map from devices to runtime.
    (TODO temporary memory)
    """

    def __init__(self, topology, device_speeds):
        self._topology = topology
        self._device_speeds = device_speeds
        self._op_register = {
            "Allreduce": self._infer_costs_for_allreduce,
            "Broadcast": self._infer_costs_for_broadcast_scatter,
            "Concat": self._infer_costs_for_concat,
            "Gather": self._infer_costs_for_gather,
            "Loss": self._infer_costs_for_loss,
            "LossGrad": self._infer_costs_for_loss_grad,
            "MatMul": self._infer_costs_for_matmul,
            "MatMulGrad": self._infer_costs_for_matmul_grad,
            "Scatter": self._infer_costs_for_broadcast_scatter,
        }

    def _infer_costs_for_allreduce(self, op, inputs, outputs):
        costs = {}
        input_size = inputs[0].type.types[0].size()
        devices = list(utils.get_all_devices(outputs))
        num_devices = len(devices)
        per_device_data = 2 * input_size * (num_devices - 1) / num_devices
        per_device_data_gb = per_device_data / BYTES_IN_GB
        all_bandwidths = []
        for i in range(len(devices)):
            for j in range(i, len(devices)):
                all_bandwidths.append(
                    self._topology.get_bandwidth(devices[i], devices[j])
                )
        average_bandwidth = np.mean(all_bandwidths)
        cost = per_device_data_gb / average_bandwidth
        for device in devices:
            costs[device] = cost

        return costs

    def _infer_costs_for_concat(self, op, inputs, outputs):
        costs = {}
        input_devices = utils.get_all_devices(inputs)
        for device in input_devices:
            # TODO: Compute cost properly
            costs[device] = 0
        return costs

    def _infer_costs_for_gather(self, op, inputs, outputs):
        costs = {}
        output_devices = utils.get_all_devices(outputs)
        for device in output_devices:
            # TODO: Compute cost properly
            costs[device] = 0

        return costs

    def _infer_costs_for_loss(self, op, inputs, outputs):
        costs = {}
        input_devices = utils.get_all_devices(inputs)
        for device in input_devices:
            # TODO: Compute cost properly
            costs[device] = 0
        return costs

    def _infer_costs_for_loss_grad(self, op, inputs, outputs):
        costs = {}
        input_devices = utils.get_all_devices(inputs)
        for device in input_devices:
            # TODO: Compute cost properly
            costs[device] = 0
        return costs

    def _infer_costs_for_matmul(self, op, inputs, outputs):
        device = inputs[0].type.device
        # TODO: Verify all input and output devices are the same?
        # TODO: Check this cost computation
        a_matrix_shape = inputs[0].type.shape
        b_matrix_shape = inputs[1].type.shape
        flops = 2 * a_matrix_shape[1] * a_matrix_shape[0] * b_matrix_shape[1]
        # TODO: Use a better way of computing runtime from FLOPs
        runtime = flops / self._device_speeds[device.device_type]
        return {device: runtime}

    def _infer_costs_for_matmul_grad(self, op, inputs, outputs):
        costs = self._infer_costs_for_matmul(op, [inputs[0], inputs[1]], [outputs[0]])
        # TODO: Replace with more accurate estimate.
        for device in costs:
            costs[device] *= 2
        return costs

    def _infer_costs_for_broadcast_scatter(self, op, inputs, outputs):
        costs = {}
        input_device = inputs[0].type.device
        costs[input_device] = 0
        input_size = inputs[0].type.size() * inputs[0].type.dtype.size
        input_size_gb = input_size / BYTES_IN_GB
        output_devices = utils.get_all_devices(outputs)
        for output_device in output_devices:
            bandwidth = self._topology.get_bandwidth(input_device, output_device)
            transfer_time = input_size_gb / bandwidth
            # NOTE: This assumes all tensors can be sent concurrently
            # TODO: Do we need to model the link capacity?
            costs[input_device] = max(costs[input_device], transfer_time)
            if output_device != input_device:
                costs[output_device] = transfer_time

        return costs

    def infer_costs(self, op):
        inputs = op.get_in_edges()
        outputs = op.get_out_edges()

        return self._op_register[op.op_type](op, inputs, outputs)
