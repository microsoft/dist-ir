from typing import Dict, Set, Tuple

from ..ir import Device, ModuleView
from .pipeline_parallel_scheduler import PipelineParallelScheduler


class FIFOScheduler(PipelineParallelScheduler):
    def _get_next_stage_to_schedule(self, device: Device) -> Tuple[ModuleView, int]:
        return self._ready_stages[device][0]
