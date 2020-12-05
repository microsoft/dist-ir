from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import Dict, Set, Tuple

from dist_ir.ir import Module, Device, Op


class Scheduler(ABC):
    def __init__(self, num_microbatches, seed=0):
        self._num_microbatches = num_microbatches
        self._rng = random.Random(seed)

    def _prepare_ops_to_schedule(self, module, partition_map):
        """Enumerates the ops to schedule on each device across all microbatches."""
        self._ops_to_schedule = defaultdict(set)
        for i in range(self._num_microbatches):
            for op_name in module.get_ops():
                device = partition_map[op_name]
                self._ops_to_schedule[device].add((op_name, i))
        self._remaining_inputs = defaultdict(set)
        for op_name, op in module.get_ops().items():
            self._remaining_inputs[op_name] = set([e.name for e in op.get_in_edges()])
        self._available_inputs = set()
        for input_value in module.get_inputs():
            self._available_inputs.add(input_value.name)

    def _is_op_ready(self, op: Op) -> bool:
        for in_edge in op.get_in_edges():
            if in_edge.name not in self._available_inputs:
                return False
        return True

    def _get_ready_ops(self, module: Module) -> Dict[Device, Set[Tuple[str, int]]]:
        # TODO: Optimize this so we don't need to re-compute from scratch every call.
        ready_ops = defaultdict(set)
        for device in self._ops_to_schedule:
            for op_to_schedule in self._ops_to_schedule[device]:
                op = module.get_op(op_to_schedule[0])
                if self._is_op_ready(op):
                    ready_ops[device].add(op_to_schedule)
        return ready_ops

    @abstractmethod
    def _get_next_op_to_schedule(
        self, ready_ops: Dict[Device, Set[Tuple[str, int]]], device: Device
    ) -> Tuple[str, int]:
        raise NotImplementedError()

    def schedule(self, module, partition_map):
        self._prepare_ops_to_schedule(module, partition_map)
        num_scheduled_ops = 0
        total_ops_to_schedule = sum(
            [len(self._ops_to_schedule[d]) for d in self._ops_to_schedule]
        )
        schedule = []
        while num_scheduled_ops < total_ops_to_schedule:
            per_timestep_schedule = {}
            ready_ops = self._get_ready_ops(module)
            for device in self._ops_to_schedule:
                if len(ready_ops[device]) > 0:
                    # NOTE: op_to_schedule is a tuple of (op_name, microbatch)
                    # TODO: Rename this?
                    op_to_schedule = self._get_next_op_to_schedule(ready_ops, device)
                    per_timestep_schedule[device] = op_to_schedule
                    self._ops_to_schedule[device].remove(op_to_schedule)
                    num_scheduled_ops += 1
                    op = module.get_op(op_to_schedule[0])
                    for out_edge in op.get_out_edges():
                        self._available_inputs.add(out_edge.name)
            assert len(per_timestep_schedule) > 0
            schedule.append(per_timestep_schedule)
        return schedule
