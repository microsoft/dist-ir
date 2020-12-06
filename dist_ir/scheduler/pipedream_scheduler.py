from collections import defaultdict
from typing import Dict, Set, Tuple

from ..ir import Device
from .scheduler import Scheduler


class PipeDreamScheduler(Scheduler):
    def __init__(self, num_microbatches):
        Scheduler.__init__(self, num_microbatches)
        self._prev_op_types = defaultdict(lambda: "bw")

    def _get_next_op_to_schedule(self, device: Device) -> Tuple[str, int]:
        ready_ops_by_type = defaultdict(list)
        for ready_op in self._ready_ops[device]:
            # TODO: Use a more robust method to identify backwards pass ops.
            if "Grad" in ready_op[0]:
                ready_ops_by_type["bw"].append(ready_op)
            else:
                ready_ops_by_type["fw"].append(ready_op)
        last_op_type = self._prev_op_types[device]
        if last_op_type == "fw":
            if len(ready_ops_by_type["bw"]) > 0:
                next_op_type = "bw"
            else:
                next_op_type = "fw"
        elif last_op_type == "bw":
            if len(ready_ops_by_type["fw"]) > 0:
                next_op_type = "fw"
            else:
                next_op_type = "bw"
        self._prev_op_types[device] = next_op_type
        # TODO: Use a more robust method for selecting the next op to run.
        sorted_ready_ops_by_type = sorted(
            ready_ops_by_type[next_op_type], key=lambda x: x[1]
        )
        return sorted_ready_ops_by_type[0]
