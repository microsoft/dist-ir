from .device import Device


class Topology:
    def __init__(self):
        self._device_id_counter = 0
        self._devices = []
        self._bandwidths = {}

    @property
    def devices(self):
        return self._devices

    # TODO: Move throughput and dram_bandwidth to common constants file
    def add_device(self, device_type, throughput=1.0e14, dram_bandwidth=1.2e12):
        device_id = self._device_id_counter
        self._device_id_counter += 1
        device = Device(device_id, device_type, throughput, dram_bandwidth)
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
