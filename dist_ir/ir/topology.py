from .device import Device
from dist_ir.utils import constants


class Topology:
    def __init__(self):
        self._device_id_counter = 0
        self._devices = []
        self._bandwidths = {}

    @property
    def devices(self):
        return self._devices

    def add_device(
        self,
        device_type,
        throughput=constants.DEFAULT_DEVICE_THROUGHPUT,
        dram_bandwidth=constants.DEFAULT_DRAM_BANDWIDTH,
        kernel_launch_overhead=constants.DEFAULT_KERNEL_LAUNCH_OVERHEAD,
    ):
        device_id = self._device_id_counter
        self._device_id_counter += 1
        device = Device(
            device_id, device_type, throughput, dram_bandwidth, kernel_launch_overhead
        )
        self._devices.append(device)
        self._bandwidths[device] = {}
        return device

    def set_bandwidth(self, device_a: Device, device_b: Device, bandwidth: float):
        """Sets the bandwidth between two devices in Gbps."""
        # TODO: Support different bandwidths in each direction?
        self._bandwidths[device_a][device_b] = bandwidth
        self._bandwidths[device_b][device_a] = bandwidth

    def get_bandwidth(self, device_a: Device, device_b: Device) -> float:
        if device_a not in self._bandwidths:
            raise ValueError(f"Invalid device {device_a}")
        elif device_a == device_b:
            return float("inf")
        elif device_b not in self._bandwidths[device_a]:
            raise ValueError(f"Bandwidth between {device_a} and {device_b} unknown")
        return self._bandwidths[device_a][device_b]


def get_uniform_topology(
    world_size,
    device_throughput=constants.DEFAULT_DEVICE_THROUGHPUT,
    dram_bandwidth=constants.DEFAULT_DRAM_BANDWIDTH,
    kernel_launch_overhead=constants.DEFAULT_KERNEL_LAUNCH_OVERHEAD,
    network_bandwidth=constants.DEFAULT_NETWORK_BANDWIDTH,
):
    topology = Topology()
    d0 = topology.add_device("gpu")
    for i in range(1, world_size + 1):
        topology.add_device(
            "gpu",
            throughput=device_throughput,
            dram_bandwidth=dram_bandwidth,
            kernel_launch_overhead=kernel_launch_overhead,
        )

    if isinstance(network_bandwidth, list):
        for (src_rank, dst_rank, bandwidth) in network_bandwidth:
            topology.set_bandwidth(
                topology.devices[src_rank], topology.devices[dst_rank], bandwidth
            )
    elif isinstance(network_bandwidth, float):
        for i in range(1, world_size + 1):
            for j in range(0, i):
                topology.set_bandwidth(
                    topology.devices[i], topology.devices[j], network_bandwidth
                )
    else:
        raise ValueError(f"Invalid network bandwidth {network_bandwidth}")
    return topology
