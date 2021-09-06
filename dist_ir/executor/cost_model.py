import numpy as np
from functools import reduce
from operator import mul

from ..ir.type import Float32, Float64, Int64, Tensor, TupleType

BYTES_IN_Gb = 1.25e8
KERNEL_LAUNCH_OVERHEAD = 10e-6


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
            ("Add", (Tensor, type(Float32()))): self._elementwise_cost_fn,
            ("Cast", (Tensor,)): self._elementwise_cost_fn,
            ("Cast", (type(Float64()),)): lambda op, x: {},
            ("Cast", (type(Int64()),)): lambda op, x: {},
            ("Concat", (Tensor, Tensor)): self._concat_cost_fn,
            ("Concat", (Tensor, Tensor, Tensor)): self._concat_cost_fn,
            ("Concat", (Tensor, Tensor, Tensor, Tensor)): self._concat_cost_fn,
            ("Constant", ()): lambda op: {},
            ("ConstantOfShape", (Tensor,)): self._constant_of_shape_cost_fn,
            ("Div", (type(Int64()), type(Int64()))): lambda op, x, y: {},
            ("Div", (Tensor, type(Float32()))): self._elementwise_cost_fn,
            ("Div", (Tensor, Tensor)): self._elementwise_cost_fn,
            ("Gather", (Tensor, type(Int64()))): self._gather_cost_fn,
            ("Gather", (Tensor, Tensor)): self._gather_cost_fn,
            ("Gemm", (Tensor, Tensor, Tensor)): self._gemm_cost_fn,
            ("Identity", (Tensor,)): self._identity_cost_fn,
            ("Join", (Tensor, Tensor)): self._join_cost_fn,
            ("Join", (Tensor, Tensor, Tensor, Tensor)): self._join_cost_fn,
            ("NonZero", (Tensor,)): self._nonzero_cost_fn,
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
            ("Mul", (Tensor, Tensor)): self._elementwise_cost_fn,
            ("Mul", (Tensor, type(Float32()))): self._elementwise_cost_fn,
            ("Mul", (type(Int64()), type(Int64()))): lambda op, x, y: {},
            ("Pow", (Tensor, type(Float32()))): self._elementwise_cost_fn,
            ("ReduceMean", (Tensor,)): self._reduce_mean_cost_fn,
            ("Relu", (Tensor,)): self._elementwise_cost_fn,
            ("ReluGrad", (Tensor, Tensor)): self._elementwise_cost_fn,
            ("Reshape", (Tensor, Tensor)): self._reshape_cost_fn,
            ("Select", (TupleType,)): self._select_cost_fn,
            ("Send", (Tensor,)): self._send_cost_fn,
            ("Send", (type(Int64()),)): lambda op, x: {},
            ("SGDOptimizer", tuple(Tensor for i in range(4))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(8))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(16))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(32))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(64))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(128))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(256))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(512))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(1024))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(2048))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(4096))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(8192))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(16384))): self._sgd_cost_fn,
            ("SGDOptimizer", tuple(Tensor for i in range(32768))): self._sgd_cost_fn,
            ("Shape", (Tensor,)): self._shape_cost_fn,
            ("Slice", (Tensor, Tensor, Tensor, Tensor)): self._slice_cost_fn,
            (
                "Slice",
                (Tensor, Tensor, Tensor, Tensor, type(Int64())),
            ): self._slice_cost_fn,
            ("Split", (Tensor,)): self._split_cost_fn,
            ("SplitUniform", (Tensor,)): self._split_cost_fn,
            ("SplitUniformToTupleType", (Tensor,)): self._split_cost_fn,
            ("Softmax", (Tensor,)): self._softmax_cost_fn,
            ("Sqrt", (Tensor,)): self._elementwise_cost_fn,
            ("Squeeze", (Tensor,)): self._squeeze_cost_fn,
            ("Sub", (type(Float32()), Tensor)): lambda op, x, y: {},
            ("Sub", (Tensor, Tensor)): self._elementwise_cost_fn,
            ("Sub", (type(Int64()), type(Int64()))): lambda op, x, y: {},
            ("Tanh", (Tensor,)): self._elementwise_cost_fn,
            ("Transpose", (Tensor,)): self._transpose_cost_fn,
            ("Unsqueeze", (type(Int64()),)): self._unsqueeze_cost_fn,
            ("Unsqueeze", (Tensor,)): self._unsqueeze_cost_fn,
        }

    def _elementwise_cost_fn(self, op, x, y=None):
        # if x.device is None:
        #    return {}
        n = reduce(mul, (x.shape[i] for i in range(len(x.shape))))
        data_size = x.dtype.size() * n
        if y is not None:
            data_size *= 2
        flops = n
        communication_cost = data_size / x.device.dram_bandwidth
        computation_cost = flops / x.device.throughput
        latency = KERNEL_LAUNCH_OVERHEAD + communication_cost + computation_cost
        return {x.device: latency}

    def _concat_cost_fn(self, op, *xs):
        # TODO: Compute cost properly
        devices = [x.device for x in xs]
        return {device: KERNEL_LAUNCH_OVERHEAD for device in devices}

    def _constant_of_shape_cost_fn(self, op, x):
        return {x.device: KERNEL_LAUNCH_OVERHEAD}

    def _gather_cost_fn(self, op, x, y):
        # TODO: Compute cost properly
        return {x.device: KERNEL_LAUNCH_OVERHEAD}

    def _gemm_cost_fn(self, op, x, y, z):
        gemm_costs = self._matmul_cost_fn(op, x, y)
        p = Tensor(shape=(x.shape[0], y.shape[1]), dtype=x.dtype, device=x.device)
        add_costs = self._elementwise_cost_fn(op, p, z)
        for d in gemm_costs:
            gemm_costs[d] += add_costs[d]
        return gemm_costs

    def _identity_cost_fn(self, op, x):
        # TODO: Compute cost properly
        return {x.device: 0}

    def _join_cost_fn(self, op, *xs):
        return {x.device: 0 for x in xs}

    def _matmul_cost_fn(self, op, x, y):
        data_size = x.dtype.size() * (x.shape[0] * x.shape[1] + y.shape[0] * y.shape[1])
        flops = 2 * x.shape[0] * x.shape[1] * y.shape[1]
        communication_cost = data_size / x.device.dram_bandwidth
        computation_cost = flops / x.device.throughput
        latency = KERNEL_LAUNCH_OVERHEAD + communication_cost + computation_cost
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
        average_input_size = np.mean([x.size() for x in xs]) * xs[0].dtype.size()
        per_device_data = 2 * average_input_size * (len(devices) - 1) / len(devices)
        per_device_data_gb = per_device_data / BYTES_IN_Gb
        cost = per_device_data_gb / average_bandwidth
        return {device: cost for device in devices}

    def _mpi_allreduce_cost_fn(self, op, *xs):
        input_size = xs[0].size()
        devices = [x.device for x in xs]
        num_devices = len(devices)
        per_device_data_gb = (2 * input_size / BYTES_IN_Gb / num_devices) * (
            num_devices - 1
        )
        # 2 * input_size * (num_devices - 1) / num_devices
        # per_device_data_gb = per_device_data / BYTES_IN_Gb
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
            input_size = x.size() * x.dtype.size()
            input_size_gb = input_size / BYTES_IN_Gb
            bandwidth = self._topology.get_bandwidth(x.device, output_device)
            transfer_time = input_size_gb / bandwidth
            costs[x.device] = transfer_time
            costs[output_device] = max(costs[output_device], transfer_time)
        return costs

    def _mpi_reduce_cost_fn(self, op, *xs):
        input_size = xs[0].size() * xs[0].dtype.size()
        input_size_gb = input_size / BYTES_IN_Gb
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

    def _nonzero_cost_fn(self, op, x):
        return {x.device: KERNEL_LAUNCH_OVERHEAD}

    def _reduce_mean_cost_fn(self, op, x):
        # TODO: Repace with more accurate function?
        return self._elementwise_cost_fn(op, x)

    def _reshape_cost_fn(self, op, x, y):
        return {x.device: KERNEL_LAUNCH_OVERHEAD}

    def _select_cost_fn(self, op, xs):
        costs = {}
        for typ in xs.types:
            costs[typ.device] = 0
        return costs

    def _send_cost_fn(self, op, x):
        costs = {}
        input_device = x.device
        # TODO send is synchronous; input device should do same work too
        input_size = x.size()
        input_size_gb = input_size / BYTES_IN_Gb
        output_device = op.attributes["device"]
        bandwidth = self._topology.get_bandwidth(input_device, output_device)
        transfer_time = input_size_gb / bandwidth
        # NOTE: This assumes all tensors can be sent concurrently
        # TODO: Do we need to model the link capacity?
        costs[input_device] = transfer_time
        costs[output_device] = transfer_time

        return costs

    def _sgd_cost_fn(self, op, *xs):
        weights = xs[: (len(xs) // 2)]
        gradients = xs[(len(xs) // 2) :]
        costs = {}
        for w, dw in zip(weights, gradients):
            costs.update(self._elementwise_cost_fn(op, w, dw))
        return costs

    def _shape_cost_fn(self, op, x):
        return {x.device: KERNEL_LAUNCH_OVERHEAD}

    def _slice_cost_fn(self, op, x, starts, ends, axes, steps=None):
        return {x.device: KERNEL_LAUNCH_OVERHEAD}  # TODO is this accurate?

    def _softmax_cost_fn(self, op, x):
        # TODO: Repace with more accurate function?
        return self._elementwise_cost_fn(op, x)

    def _split_cost_fn(self, op, x):
        return {x.device: KERNEL_LAUNCH_OVERHEAD}

    def _squeeze_cost_fn(self, op, x):
        return {x.device: KERNEL_LAUNCH_OVERHEAD}

    def _transpose_cost_fn(self, op, x):
        # TODO: Repace with more accurate function?
        return self._elementwise_cost_fn(op, x)

    def _unsqueeze_cost_fn(self, op, x):
        return {x.device: KERNEL_LAUNCH_OVERHEAD}
