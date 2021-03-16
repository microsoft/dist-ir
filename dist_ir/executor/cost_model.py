import numpy as np

from ..ir.type import Tensor, TupleType

BYTES_IN_GB = 8.0e9
KERNEL_LAUNCH_OVERHEAD = 1.0e-6


class CostModel:
    """A cost model -- shape-based analytical cost functions for each op type.

    These cost functions expect as input the type of each input value and output
    a map from devices to runtime.
    (TODO temporary memory)

    Cost functions don't need to check types or devices of inputs, these are
    checked by type prop functions.
    """

    # TODO instead of passing the op, should we pass the attributes as kwargs?

    def __init__(self, topology):
        self._topology = topology

        def notImplemented(*args):
            raise NotImplementedError

        # TODO: Add support for variadic inputs
        self.cost_functions = {
            ("Add", (Tensor, Tensor)): self._elementwise_cost_fn,
            ("Cast", (Tensor,)): self._cast_cost_fn,
            ("Concat", (Tensor, Tensor)): self._concat_cost_fn,
            ("Identity", (Tensor,)): self._identity_cost_fn,
            ("Join", (Tensor, Tensor)): self._join_cost_fn,
            ("Join", (Tensor, Tensor, Tensor, Tensor)): self._join_cost_fn,
            ("MPIAllgather", (Tensor,) * 2): self._mpi_allgather_cost_fn,
            ("MPIAllgather", (Tensor,) * 4): self._mpi_allgather_cost_fn,
            ("MPIAllgather", (Tensor,) * 8): self._mpi_allgather_cost_fn,
            ("MPIAllgather", (Tensor,) * 16): self._mpi_allgather_cost_fn,
            ("MPIAllgather", (Tensor,) * 32): self._mpi_allgather_cost_fn,
            ("MPIAllgather", (Tensor,) * 64): self._mpi_allgather_cost_fn,
            ("MPIAllgather", (Tensor,) * 128): self._mpi_allgather_cost_fn,
            ("MPIAllgather", (Tensor,) * 256): self._mpi_allgather_cost_fn,
            ("MPIAllgather", (Tensor,) * 512): self._mpi_allgather_cost_fn,
            ("MPIAllgather", (Tensor,) * 1024): self._mpi_allgather_cost_fn,
            ("MPIAllgather", (Tensor,) * 2048): self._mpi_allgather_cost_fn,
            ("MPIAllgather", (Tensor,) * 4096): self._mpi_allgather_cost_fn,
            ("MPIAllgather", (Tensor,) * 8192): self._mpi_allgather_cost_fn,
            ("MPIAllreduce", (Tensor,) * 2): self._mpi_allreduce_cost_fn,
            ("MPIAllreduce", (Tensor,) * 4): self._mpi_allreduce_cost_fn,
            ("MPIAllreduce", (Tensor,) * 8): self._mpi_allreduce_cost_fn,
            ("MPIAllreduce", (Tensor,) * 16): self._mpi_allreduce_cost_fn,
            ("MPIAllreduce", (Tensor,) * 32): self._mpi_allreduce_cost_fn,
            ("MPIAllreduce", (Tensor,) * 64): self._mpi_allreduce_cost_fn,
            ("MPIAllreduce", (Tensor,) * 128): self._mpi_allreduce_cost_fn,
            ("MPIAllreduce", (Tensor,) * 256): self._mpi_allreduce_cost_fn,
            ("MPIAllreduce", (Tensor,) * 512): self._mpi_allreduce_cost_fn,
            ("MPIAllreduce", (Tensor,) * 1024): self._mpi_allreduce_cost_fn,
            ("MPIAllreduce", (Tensor,) * 2048): self._mpi_allreduce_cost_fn,
            ("MPIAllreduce", (Tensor,) * 4096): self._mpi_allreduce_cost_fn,
            ("MPIAllreduce", (Tensor,) * 8192): self._mpi_allreduce_cost_fn,
            ("MPIBroadcast", (Tensor,)): self._mpi_broadcast_cost_fn,
            ("MPIBroadcastToTupleType", (Tensor,)): self._mpi_broadcast_cost_fn,
            ("MPIGather", (Tensor,) * 2): self._mpi_gather_cost_fn,
            ("MPIGather", (Tensor,) * 4): self._mpi_gather_cost_fn,
            ("MPIGather", (Tensor,) * 8): self._mpi_gather_cost_fn,
            ("MPIGather", (Tensor,) * 16): self._mpi_gather_cost_fn,
            ("MPIGather", (Tensor,) * 32): self._mpi_gather_cost_fn,
            ("MPIGather", (Tensor,) * 64): self._mpi_gather_cost_fn,
            ("MPIGather", (Tensor,) * 128): self._mpi_gather_cost_fn,
            ("MPIGather", (Tensor,) * 256): self._mpi_gather_cost_fn,
            ("MPIGather", (Tensor,) * 512): self._mpi_gather_cost_fn,
            ("MPIGather", (Tensor,) * 1024): self._mpi_gather_cost_fn,
            ("MPIGather", (Tensor,) * 2048): self._mpi_gather_cost_fn,
            ("MPIGather", (Tensor,) * 4096): self._mpi_gather_cost_fn,
            ("MPIGather", (Tensor,) * 8192): self._mpi_gather_cost_fn,
            (
                "MPIGatherFromTupleType",
                (TupleType,),
            ): lambda op, xs: self._mpi_gather_cost_fn(op, *xs.types),
            ("MPIReduce", (Tensor,) * 2): self._mpi_reduce_cost_fn,
            ("MPIReduce", (Tensor,) * 4): self._mpi_reduce_cost_fn,
            ("MPIReduce", (Tensor,) * 8): self._mpi_reduce_cost_fn,
            ("MPIReduce", (Tensor,) * 16): self._mpi_reduce_cost_fn,
            ("MPIReduce", (Tensor,) * 32): self._mpi_reduce_cost_fn,
            ("MPIReduce", (Tensor,) * 64): self._mpi_reduce_cost_fn,
            ("MPIReduce", (Tensor,) * 128): self._mpi_reduce_cost_fn,
            ("MPIReduce", (Tensor,) * 256): self._mpi_reduce_cost_fn,
            ("MPIReduce", (Tensor,) * 512): self._mpi_reduce_cost_fn,
            ("MPIReduce", (Tensor,) * 1024): self._mpi_reduce_cost_fn,
            ("MPIReduce", (Tensor,) * 2048): self._mpi_reduce_cost_fn,
            ("MPIReduce", (Tensor,) * 4096): self._mpi_reduce_cost_fn,
            ("MPIReduce", (Tensor,) * 8192): self._mpi_reduce_cost_fn,
            ("MPIScatter", (Tensor,)): self._mpi_scatter_cost_fn,
            ("MPIScatterToTupleType", (Tensor,)): self._mpi_scatter_cost_fn,
            # ("MPIAllreduce_v2", (TupleType,)): self._allreduce_cost_fn,
            ("Loss", (Tensor, Tensor)): self._elementwise_cost_fn,
            ("LossGrad", (Tensor, Tensor)): self._elementwise_cost_fn,
            ("MatMul", (Tensor, Tensor)): self._matmul_cost_fn,
            ("MatMulGrad", (Tensor, Tensor, Tensor)): self._matmul_grad_cost_fn,
            ("Min", (Tensor, Tensor)): self._min_cost_fn,
            ("Relu", (Tensor,)): self._elementwise_cost_fn,
            ("ReluGrad", (Tensor, Tensor)): self._elementwise_cost_fn,
            ("Select", (TupleType,)): self._select_cost_fn,
            ("Send", (Tensor,)): self._send_cost_fn,
            ("Split", (Tensor,)): self._split_cost_fn,
            ("Shape", (Tensor,)): self._shape_cost_fn,
            ("Slice", (Tensor, Tensor, Tensor, Tensor)): self._slice_cost_fn,
        }

    def _elementwise_cost_fn(self, op, x, y=None):
        flops = x.size()
        runtime = flops / x.device.throughput
        return {x.device: runtime}

    def _cast_cost_fn(self, op, x):
        return {x.device: x.size()}

    def _concat_cost_fn(self, op, *xs):
        # TODO: Compute cost properly
        devices = [x.device for x in xs]
        return {device: 0 for device in devices}

    def _identity_cost_fn(self, op, x):
        # TODO: Compute cost properly
        return {x.device: 0}

    def _join_cost_fn(self, op, *xs):
        return {x.device: 0 for x in xs}

    def _matmul_cost_fn(self, op, x, y):
        data_size = 2 * (x.shape[0] * x.shape[1] + y.shape[0] * y.shape[1])
        flops = 2 * x.shape[0] * x.shape[1] * y.shape[1]
        communication_cost = data_size / x.device.dram_bandwidth
        computation_cost = flops / x.device.throughput
        latency = communication_cost + computation_cost
        return {x.device: latency}

    def _matmul_grad_cost_fn(self, op, x, y, dz):
        # dx = dz * y.T, dy = x.T * dz
        xT = Tensor(dtype=x.dtype, shape=(x.shape[1], x.shape[0]), device=x.device)
        yT = Tensor(dtype=y.dtype, shape=(y.shape[1], y.shape[0]), device=y.device)
        costs1 = self._matmul_cost_fn(op, dz, yT)
        costs2 = self._matmul_cost_fn(op, xT, dz)
        return {x.device: costs1[x.device] + costs2[x.device]}

    def _min_cost_fn(self, op, x, y):
        return {x.device: x.size()}

    def _mpi_allgather_cost_fn(self, op, *xs):
        # TODO: Verify correctness
        devices = [x.device for x in xs]
        all_bandwidths = []
        for i in range(len(devices)):
            for j in range(i + 1, len(devices)):
                all_bandwidths.append(
                    self._topology.get_bandwidth(devices[i], devices[j])
                )
        average_bandwidth = np.mean(all_bandwidths)
        average_input_size = np.mean([x.size() for x in xs]) * xs[0].dtype.size
        per_device_data = 2 * average_input_size * (len(devices) - 1) / len(devices)
        per_device_data_gb = per_device_data / BYTES_IN_GB
        cost = per_device_data_gb / average_bandwidth
        return {device: cost for device in device}

    def _mpi_allreduce_cost_fn(self, op, *xs):
        input_size = xs[0].size()
        devices = [x.device for x in xs]
        num_devices = len(devices)
        per_device_data = 2 * input_size * (num_devices - 1) / num_devices
        per_device_data_gb = per_device_data / BYTES_IN_GB
        all_bandwidths = []
        for i in range(len(devices)):
            for j in range(i + 1, len(devices)):
                all_bandwidths.append(
                    self._topology.get_bandwidth(devices[i], devices[j])
                )
        average_bandwidth = np.mean(all_bandwidths)
        cost = per_device_data_gb / average_bandwidth

        return {device: cost for device in devices}

    def _mpi_broadcast_cost_fn(self, op, x):
        cost = 0
        # cost = x.size()
        return {d: cost for d in op.attributes["devices"]}

    def _mpi_gather_cost_fn(self, op, *xs):
        output_device = op.attributes["device"]
        costs = {output_device: 0}
        for x in xs:
            input_size = x.size() * x.dtype.size
            input_size_gb = input_size / BYTES_IN_GB
            bandwidth = self._topology.get_bandwidth(x.device, output_device)
            transfer_time = input_size_gb / bandwidth
            costs[x.device] = transfer_time
            costs[output_device] = max(costs[output_device], transfer_time)
        return costs

    def _mpi_reduce_cost_fn(self, op, *xs):
        input_size = xs[0].size() * xs[0].dtype.size
        input_size_gb = input_size / BYTES_IN_GB
        output_device = op.attributes["device"]
        costs = {output_device: 0}
        for x in xs:
            bandwidth = self._topology.get_bandwidth(x.device, output_device)
            transfer_time = input_size_gb / bandwidth
            costs[x.device] = transfer_time
            costs[output_device] = max(costs[output_device], transfer_time)
        return costs

    def _mpi_scatter_cost_fn(self, op, x):
        # cost = x.size()
        cost = 0
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
        input_size = x.size() * x.dtype.size
        input_size_gb = input_size / BYTES_IN_GB
        output_device = op.attributes["device"]
        bandwidth = self._topology.get_bandwidth(input_device, output_device)
        transfer_time = input_size_gb / bandwidth
        # NOTE: This assumes all tensors can be sent concurrently
        # TODO: Do we need to model the link capacity?
        costs[input_device] = transfer_time
        costs[output_device] = transfer_time

        return costs

    def _shape_cost_fn(self, op, x):
        return {x.device: 0}

    def _slice_cost_fn(self, op, x, starts, ends, axes):
        return {x.device: KERNEL_LAUNCH_OVERHEAD}  # TODO is this accurate?

    def _split_cost_fn(self, op, x):
        return {x.device: KERNEL_LAUNCH_OVERHEAD}
