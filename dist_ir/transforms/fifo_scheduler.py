from typing import Dict, Set, Tuple

from ..ir import Device
from .pipeline_parallel_scheduler import PipelineParallelScheduler


class FIFOScheduler(PipelineParallelScheduler):
    def _get_next_op_to_schedule(self, device: Device) -> Tuple[str, int]:
        return self._ready_ops[device][0]
