from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Set, Tuple

from dist_ir.ir import Module, Device, Op


class Scheduler(ABC):
    def __init__(self, num_microbatches):
        self._num_microbatches = num_microbatches

    def _prepare_ops_to_schedule(self, module, partition_map):
        """Enumerates the ops to schedule on each device across all microbatches."""
        self._remaining_inputs = defaultdict(lambda: 0)
        self._ready_ops = defaultdict(list)
        for op_name, op in module.get_ops().items():
            device = partition_map[op_name]
            in_edges = op.get_in_edges()
            remaining_inputs = len(in_edges)
            for in_edge in in_edges:
                if module.is_input(in_edge.name):
                    remaining_inputs -= 1
            for i in range(self._num_microbatches):
                self._remaining_inputs[(op_name, i)] = remaining_inputs
                if remaining_inputs == 0:
                    self._ready_ops[device].append((op_name, i))

    @abstractmethod
    def _get_next_op_to_schedule(self, device: Device) -> Tuple[str, int]:
        raise NotImplementedError()

    def schedule(self, module, partition_map):
        self._prepare_ops_to_schedule(module, partition_map)
        num_scheduled_ops = 0
        total_ops_to_schedule = len(module.get_ops()) * self._num_microbatches
        schedule = []
        while num_scheduled_ops < total_ops_to_schedule:
            per_timestep_schedule = {}
            devices = list(self._ready_ops.keys())
            for device in devices:
                if len(self._ready_ops[device]) > 0:
                    op_to_schedule = self._get_next_op_to_schedule(device)
                    (op_name, microbatch) = op_to_schedule
                    per_timestep_schedule[device] = op_to_schedule
                    # TODO: Optimize this so it isn't an O(N) call?
                    self._ready_ops[device].remove(op_to_schedule)
                    num_scheduled_ops += 1
                    op = module.get_op(op_name)
                    for out_edge in op.get_out_edges():
                        consumers = module.get_consumers_for_value(out_edge.name)
                        for consumer in consumers:
                            consumer_op = (consumer, microbatch)
                            self._remaining_inputs[consumer_op] -= 1
                            if self._remaining_inputs[consumer_op] == 0:
                                consumer_op_device = partition_map[consumer]
                                self._ready_ops[consumer_op_device].append(consumer_op)
            if len(per_timestep_schedule) == 0:
                raise RuntimeError(
                    f"No ops to schedule in iteration {len(schedule) + 1}"
                )
            schedule.append(per_timestep_schedule)
        return schedule
