import numpy as np

from ..ir.type import Tensor, TupleType

BYTES_IN_GB = 8.0e9


class CostModel:
    """A cost model -- shape-based analytical cost functions for each op type.

    These cost functions expect as input the type of each input value and output
    a map from devices to runtime.
    (TODO temporary memory)

    Cost functions don't need to check types or devices of inputs, these are
    checked by type prop functions.
    """

    # TODO instead of passing the op, should we pass the attributes as kwargs?

    def __init__(self, topology, device_speeds):
        self._topology = topology
        # TODO shouldn't device speeds be part of the topology?
        self._device_speeds = device_speeds

        def notImplemented(*args):
            raise NotImplementedError

        self.cost_functions = {
            ("Add", (Tensor, Tensor)): self._add_cost_fn,
            ("Allreduce", (TupleType,)): self._allreduce_cost_fn,
            ("Broadcast", (Tensor,)): self._broadcast_cost_fn,
            ("Cast", (Tensor,)): self._cast_cost_fn,
            ("Concat", (TupleType,)): self._concat_cost_fn,
            ("Identity", (Tensor,)): self._identity_cost_fn,
            ("Join", (Tensor, Tensor)): self._join_cost_fn,
            ("Join", (Tensor, Tensor, Tensor, Tensor)): self._join_cost_fn,
            ("MPIGather", (TupleType,)): self._mpi_gather_cost_fn,
            ("MPIReduce", (TupleType,)): self._mpi_reduce_cost_fn,
            ("Loss", (Tensor, Tensor)): self._loss_cost_fn,
            ("LossGrad", (Tensor, Tensor)): self._loss_grad_cost_fn,
            ("MatMul", (Tensor, Tensor)): self._matmul_cost_fn,
            ("MatMulGrad", (Tensor, Tensor, Tensor)): self._matmul_grad_cost_fn,
            ("Min", (Tensor, Tensor)): self._min_cost_fn,
            ("Relu", (Tensor,)): self._relu_cost_fn,
            ("Relu", (Tensor, Tensor)): self._relu_grad_cost_fn,
            ("Scatter", (Tensor,)): self._scatter_cost_fn,
            ("Select", (TupleType,)): self._select_cost_fn,
            ("Send", (Tensor,)): self._send_cost_fn,
            ("Split", (Tensor,)): notImplemented,
            ("Shape", (Tensor,)): self._shape_cost_fn,
            ("Slice", (Tensor, Tensor, Tensor, Tensor)): self._slice_cost_fn,
        }

    def _add_cost_fn(self, op, x, y):
        # TODO: Check this cost computation
        flops = x.size()
        # TODO: Use a better way of computing runtime from FLOPs
        runtime = flops / self._device_speeds[x.device.device_type]
        return {x.device: runtime}

    def _allreduce_cost_fn(self, op, xs):
        input_size = xs.types[0].size()
        devices = list(xs.get_all_devices())
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

        return {device: cost for device in devices}

    def _broadcast_cost_fn(self, op, x):
        cost = x.size()
        return {d: cost for d in op.attributes["devices"]}

    def _cast_cost_fn(self, op, x):
        return {x.device: x.size()}

    def _concat_cost_fn(self, op, xs):
        # TODO: Compute cost properly
        devices = xs.get_all_devices()
        return {device: 0 for device in devices}

    def _mpi_gather_cost_fn(self, op, xs):
        # TODO: Compute cost properly
        devices = xs.get_all_devices()
        return {device: 0 for device in devices}

    def _mpi_reduce_cost_fn(self, op, xs):
        devices = xs.get_all_devices()
        return {device: 0 for device in devices}

    def _identity_cost_fn(self, op, x):
        # TODO: Compute cost properly
        return {x.device: 0}

    def _join_cost_fn(self, op, *xs):
        costs = {}
        for x in xs:
            costs[x.device] = 0
        return costs

    def _loss_cost_fn(self, op, x, y):
        # TODO: Compute cost properly
        return {x.device: 0}

    def _loss_grad_cost_fn(self, op, x, y):
        # TODO: Compute cost properly
        return {x.device: 0}

    def _matmul_cost_fn(self, op, x, y):
        # TODO integrate device speed
        return {x.device: 2 * x.shape[0] * x.shape[1] * y.shape[1]}

    def _matmul_cost_fn(self, op, x, y):
        # TODO: Check this cost computation
        flops = 2 * x.shape[0] * x.shape[1] * y.shape[1]
        # TODO: Use a better way of computing runtime from FLOPs
        runtime = flops / self._device_speeds[x.device.device_type]
        return {x.device: runtime}

    def _matmul_grad_cost_fn(self, op, x, y, dz):
        # TODO: Check this cost computation
        # dx = dz * y.T, dy = x.T * dz
        xT = Tensor(dtype=x.dtype, shape=(x.shape[1], x.shape[0]), device=x.device)
        yT = Tensor(dtype=y.dtype, shape=(y.shape[1], y.shape[0]), device=y.device)
        costs1 = self._matmul_cost_fn(op, dz, yT)
        costs2 = self._matmul_cost_fn(op, xT, dz)
        return {x.device: costs1[x.device] + costs2[x.device]}

    def _min_cost_fn(self, op, x, y):
        return {x.device: x.size()}

    def _relu_cost_fn(self, op, x):
        return {x.device: 0}

    def _relu_grad_cost_fn(self, op, x, y):
        return {x.device: 0}

    def _scatter_cost_fn(self, op, x):
        cost = x.size()
        return {d: cost for d in op.attributes["devices"]}

    def _select_cost_fn(self, op, xs):
        costs = {}
        for typ in xs.types:
            costs[typ.device] = 0
        return costs

    def _send_cost_fn(self, op, x):
        costs = {}
        input_device = x.device
        # TODO send is synchronous; input device should do same work too
        costs[input_device] = 0
        input_size = x.size() * x.dtype.size
        input_size_gb = input_size / BYTES_IN_GB
        output_device = op.attributes["device"]
        bandwidth = self._topology.get_bandwidth(input_device, output_device)
        transfer_time = input_size_gb / bandwidth
        # NOTE: This assumes all tensors can be sent concurrently
        # TODO: Do we need to model the link capacity?
        costs[input_device] = max(costs[input_device], transfer_time)
        if output_device != input_device:
            costs[output_device] = transfer_time

        return costs

    def _shape_cost_fn(self, op, x):
        return {x.device: 1}  # TODO 1 clock cycle? 1 flop?

    def _slice_cost_fn(self, op, x, starts, ends, axes):
        return {x.device: 1}  # TODO is this accurate?
