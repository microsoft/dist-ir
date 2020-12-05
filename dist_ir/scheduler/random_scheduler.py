from typing import Dict, Set, Tuple

from ..ir import Device
from .scheduler import Scheduler


class RandomScheduler(Scheduler):
    def _get_next_op_to_schedule(
        self, ready_ops: Dict[Device, Set[Tuple[str, int]]], device: Device
    ) -> Tuple[str, int]:
        return self._rng.choice(sorted(ready_ops[device]))
