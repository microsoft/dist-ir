from collections import defaultdict
from typing import Dict, Set, Tuple

from ..ir import Device, ModuleView
from .pipeline_parallel_scheduler import PipelineParallelScheduler


class PipeDreamScheduler(PipelineParallelScheduler):
    def __init__(self, num_microbatches):
        PipelineParallelScheduler.__init__(self, num_microbatches)
        self._prev_stage_types = defaultdict(lambda: "bw")

    def _get_next_stage_to_schedule(self, device: Device) -> Tuple[ModuleView, int]:
        ready_stages_by_type = defaultdict(list)
        for ready_stage in self._ready_stages[device]:
            # TODO: Use a more robust method to identify backwards pass stages.
            (stage, microbatch) = ready_stage
            ops = stage.get_ops()
            if "Grad" in list(ops.keys())[0]:
                ready_stages_by_type["bw"].append(ready_stage)
            else:
                ready_stages_by_type["fw"].append(ready_stage)
        last_stage_type = self._prev_stage_types[device]
        if last_stage_type == "fw":
            if len(ready_stages_by_type["bw"]) > 0:
                next_stage_type = "bw"
            else:
                next_stage_type = "fw"
        elif last_stage_type == "bw":
            if len(ready_stages_by_type["fw"]) > 0:
                next_stage_type = "fw"
            else:
                next_stage_type = "bw"
        self._prev_stage_types[device] = next_stage_type
        return ready_stages_by_type[next_stage_type][0]
