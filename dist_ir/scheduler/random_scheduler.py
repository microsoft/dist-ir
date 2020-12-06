from typing import Dict, Set, Tuple

from ..ir import Device
from .scheduler import Scheduler


class RandomScheduler(Scheduler):
    def _get_next_op_to_schedule(self, device: Device) -> Tuple[str, int]:
        return self._rng.choice(sorted(self._ready_ops[device]))
