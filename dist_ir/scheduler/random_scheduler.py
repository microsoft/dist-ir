from typing import Dict, Set, Tuple
import random

from dist_ir.ir import Module, Device, Op
from .scheduler import Scheduler


class RandomScheduler(Scheduler):
    def _get_next_op_to_schedule(
        self, ready_ops_on_device: Dict[Device, Set[Tuple[str, int]]]
    ) -> Tuple[str, int]:
        return random.sample(ready_ops_on_device, 1)[0]
