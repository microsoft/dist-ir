from typing import List

from .device import Device


@dataclass
class DeviceNamespace(frozen=True):
    devices: Tuple[Devices]

    def partition(self, num_partitions: int) -> List["DeviceNamespace"]:
        num_devices = len(self._all_devices)
        if num_partitions < num_devices or num_partitions % num_devices > 0:
            raise ValueError(
                f"Invalid # of partitions {num_partitions} "
                f"for namespace of size {num_devices}"
            )
        devices_per_partition = num_partitions // num_devices
        return [
            DeviceNamespace(
                self.devices[
                    i * devices_per_partition : (i + 1) * devices_per_partition
                ],
            )
            for i in range(num_partitions)
        ]
