from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Set, Tuple

from ..ir import Function, Device, Op
from . import utils


class PipelineParallelScheduler(ABC):
    """Interface for a pipeline parallel scheduler.

    Pipeline parallel schedulers take as input a DistIR function, the number of
    microbatches to partition each minibatch into, and a partition map which
    captures the explicit placement of each stage onto corresponding devices.
    The scheduler will return a time-ordered list of stages to execute on each
    device.

    Each subclass must implement the _get_next_stage_to_schedule function which
    decides the next stage to schedule for each device using any necessary state.
    """

    def __init__(self, num_microbatches):
        self._num_microbatches = num_microbatches
        self._remaining_inputs = defaultdict(lambda: 0)
        self._ready_stages = defaultdict(list)

    def _prepare_stages_to_schedule(self, function, partition_map):
        """Enumerates the stages to schedule on each device across all microbatches."""
        for stage, device in partition_map.items():
            inputs = stage.inputs
            remaining_inputs = len(inputs)
            for input in inputs:
                if input in function.inputs:
                    remaining_inputs -= 1
            for i in range(self._num_microbatches):
                self._remaining_inputs[(stage.name, i)] = remaining_inputs
                if remaining_inputs == 0:
                    self._ready_stages[device].append((stage, i))

    @abstractmethod
    def _get_next_stage_to_schedule(self, device: Device) -> Tuple[Function, int]:
        raise NotImplementedError()

    def schedule(self, function, partition_map):
        # Verify that all stage names are unique
        assert len(set([stage.name for stage in partition_map])) == len(partition_map)
        self._prepare_stages_to_schedule(function, partition_map)
        op_to_stage = utils.get_op_to_stage_map(list(partition_map.keys()))
        num_scheduled_stages = 0
        total_stages_to_schedule = len(partition_map) * self._num_microbatches
        schedule = []
        while num_scheduled_stages < total_stages_to_schedule:
            next_ready_stages = []
            per_timestep_schedule = {}
            devices = list(self._ready_stages.keys())
            for device in devices:
                if len(self._ready_stages[device]) > 0:
                    stage_to_schedule = self._get_next_stage_to_schedule(device)
                    (stage, microbatch) = stage_to_schedule
                    per_timestep_schedule[device] = stage_to_schedule
                    # TODO: Optimize this so it isn't an O(N) call?
                    self._ready_stages[device].remove(stage_to_schedule)
                    num_scheduled_stages += 1
                    outputs = stage.outputs
                    for output in outputs:
                        consumer_ops = function.consumers[output]
                        consumer_stages = set([op_to_stage[op] for op in consumer_ops])
                        for consumer_stage in consumer_stages:
                            consumer_stage_key = (consumer_stage.name, microbatch)
                            self._remaining_inputs[consumer_stage_key] -= 1
                            if self._remaining_inputs[consumer_stage_key] == 0:
                                consumer_stage_device = partition_map[consumer_stage]
                                next_ready_stages.append(
                                    (consumer_stage_device, consumer_stage, microbatch)
                                )
            if len(per_timestep_schedule) == 0:
                raise RuntimeError(
                    f"No ops to schedule in iteration {len(schedule) + 1}"
                )
            schedule.append(per_timestep_schedule)
            for (
                consumer_stage_device,
                consumer_stage,
                microbatch,
            ) in next_ready_stages:
                self._ready_stages[consumer_stage_device].append(
                    (consumer_stage, microbatch)
                )

        return schedule
