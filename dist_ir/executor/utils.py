from typing import List
from ..ir.device import Device
from ..ir.value import Value


def get_all_devices(values: List[Value]) -> List[Device]:
    devices = set()
    for value in values:
        devices.update(value.type.get_all_devices())
    return list(devices)
