from collections import defaultdict
from typing import Dict, Set, Tuple

from ..ir import Device, Function
from .pipeline_parallel_scheduler import PipelineParallelScheduler


class FIFOScheduler(PipelineParallelScheduler):
    """Implements a FIFO schedule where all forward pass stages are executed before
    backward pass stages."""

    def _get_next_stage_to_schedule(self, device: Device) -> Tuple[Function, int]:
        ready_stages_by_type = defaultdict(list)
        for ready_stage in self._ready_stages[device]:
            # TODO: Use a more robust method to identify backwards pass stages.
            (stage, microbatch) = ready_stage
            ops = stage.get_ops()
            if "Grad" in list(ops.keys())[0]:
                ready_stages_by_type["bw"].append(ready_stage)
            else:
                ready_stages_by_type["fw"].append(ready_stage)
        if len(ready_stages_by_type["fw"]) > 0:
            return ready_stages_by_type["fw"][0]
        else:
            return ready_stages_by_type["bw"][0]
