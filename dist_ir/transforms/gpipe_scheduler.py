# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from typing import Dict, Set, Tuple

from ..ir import Device, Function
from .pipeline_parallel_scheduler import PipelineParallelScheduler


class GPipeScheduler(PipelineParallelScheduler):
    """Implements the forward then backward schedule from GPipe."""

    def __init__(self, num_microbatches):
        PipelineParallelScheduler.__init__(self, num_microbatches)

    def _get_next_stage_to_schedule(self, device: Device) -> Tuple[Function, int]:
        ready_stages_by_type = defaultdict(list)
        for ready_stage in self._ready_stages[device]:
            # TODO: Use a more robust method to identify backwards pass stages.
            (stage, microbatch) = ready_stage
            if "Grad" in stage.ops[0].op_type:
                ready_stages_by_type["bw"].append(ready_stage)
            else:
                ready_stages_by_type["fw"].append(ready_stage)
        if len(ready_stages_by_type["fw"]) > 0:
            return ready_stages_by_type["fw"][0]
        else:
            assert len(ready_stages_by_type["bw"]) > 0
            return ready_stages_by_type["bw"][0]
