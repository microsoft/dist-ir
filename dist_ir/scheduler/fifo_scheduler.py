from typing import Dict, Set, Tuple

from ..ir import Device
from .scheduler import Scheduler


class FIFOScheduler(Scheduler):
    def _get_next_op_to_schedule(self, device: Device) -> Tuple[str, int]:
        ready_ops = sorted(self._ready_ops[device], key=lambda x: x[1])
        return ready_ops[0]
