from dataclasses import dataclass
from typing import ClassVar

from ..proto import device_pb2


@dataclass(frozen=True)
class Device:

    device_id: str
    device_type: str
    throughput: float = 1.0e14
    dram_bandwidth: float = 1.2e12
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

    def serialize_to_proto(self):
        device_proto = device_pb2.Device()
        device_proto.device_id = self.device_id
        device_proto.device_type = self.device_type
        device_proto.throughput = self.throughput
        device_proto.dram_bandwidth = self.dram_bandwidth
        return device_proto
