from dataclasses import dataclass
from typing import ClassVar

from dist_ir.utils import constants


@dataclass(frozen=True)
class Device:

    device_id: str
    device_type: str
    throughput: float = constants.DEFAULT_DEVICE_THROUGHPUT
    dram_bandwidth: float = constants.DEFAULT_DRAM_BANDWIDTH
    kernel_launch_overhead: float = constants.DEFAULT_KERNEL_LAUNCH_OVERHEAD
    is_variable: bool = False

    device_variable_id: ClassVar[int] = 0

    def __str__(self):
        return f"{self.device_id} ({self.device_type})"

    def __lt__(self, other):
        return self.device_id < other.device_id

    @classmethod
    def get_new_device_variable(cls, device_type):
        device_id = f"d{cls.device_variable_id}"
        cls.device_variable_id += 1
        return Device(device_id, device_type, is_variable=True)
