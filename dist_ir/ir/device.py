from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Device:

    device_variable_id: ClassVar[int] = 0

    device_id: str
    device_type: str
    is_variable: bool = False

    def __str__(self):
        return f"{self.device_id} ({self.device_type})"

    def __lt__(self, other):
        return self.device_id < other.device_id

    @classmethod
    def get_new_device_variable(cls, device_type):
        device_id = f"d{cls.device_variable_id}"
        cls.device_variable_id += 1
        return Device(device_id, device_type, is_variable=True)
