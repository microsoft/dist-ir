from ..ir.module import Module
from ..ir.value import Value

import copy
from collections import defaultdict


class PipelineParallelTransform:
    """Partitions a module using pipeline parallelism.

    Attributes:
      num_microbatches: The number of microbatches per pipeline iteration.
      batch_dims: A map from input value name to partition dimension.
      reduction_ops: A map from output value name to a map of reduction op params.
      schedule: A list of maps from device to a tuple of (op_name, microbatch).
    """

    def __init__(self, num_microbatches, batch_dims, reduction_params, schedule):
        self._num_microbatches = num_microbatches
        self._batch_dims = batch_dims
        self._reduction_params = reduction_params
        self._schedule = schedule

    def _partition_inputs(self, module, transformed_module, pipelined_value_map):
        input_values = module.get_inputs()
        for input_value in input_values:
            v = transformed_module.add_input_value(
                input_value.name, copy.deepcopy(input_value.type)
            )
            if input_value.name in self._batch_dims:
                vs = transformed_module.add_op(
                    "Split",
                    name=f"Split/{v.name}",
                    inputs=[v],
                    attributes={
                        "num_splits": self._num_microbatches,
                        "split_dim": self._batch_dims[input_value.name],
                    },
                    output_names=[f"{v.name}s"],
                )
                pipelined_value_map[input_value.name] = {}
                for i in range(self._num_microbatches):
                    v_i = transformed_module.add_op(
                        "Select",
                        name=f"Select/{v.name}_{i}",
                        attributes={"dim": i},
                        inputs=[vs],
                        output_names=[f"{v.name}_{i}"],
                    )
                    pipelined_value_map[input_value.name][i] = v_i
            else:
                for i in range(self._num_microbatches):
                    pipelined_value_map[input_value.name][i] = v

    def _aggregate_outputs(
        self,
        transformed_module,
        orig_output,
        pipelined_output,
        merged_outputs,
        num_completed_microbatches,
    ):
        if self._reduction_params[orig_output.name] is None:
            return
        reduction_op_type = self._reduction_params[orig_output.name]["op_type"]
        if num_completed_microbatches == 1:
            merged_outputs[orig_output.name] = pipelined_output
        else:
            merged_output = merged_outputs[orig_output.name]
            if reduction_op_type == "Add":
                op_name = f"Add/{merged_output.name}-{pipelined_output.name}"
                output_name = f"{orig_output.name}/merged_{num_completed_microbatches}"
                merged_outputs[orig_output.name] = transformed_module.add_op(
                    "Add",
                    name=op_name,
                    inputs=[merged_output, pipelined_output],
                    output_names=[output_name],
                )
            elif reduction_op_type == "Concat":
                dim = self._reduction_params[orig_output.name]["dim"]
                op_name = f"Concat/{merged_output.name}-{pipelined_output.name}"
                output_name = f"{orig_output.name}/merged_{num_completed_microbatches}"
                merged_outputs[orig_output.name] = transformed_module.add_op(
                    "Concat",
                    attributes={"dim": dim},
                    name=op_name,
                    inputs=[merged_output, pipelined_output],
                    output_names=[output_name],
                )
            else:
                raise ValueError(
                    f"Unknown reduction op type {reduction_op_type} "
                    f"for output {orig_output}"
                )

    def apply(self, module):

        transformed_module = Module()

        # A map from original value name to another map from microbatch number to
        # pipelined value.
        pipelined_value_map = defaultdict(lambda: defaultdict(Value))

        # A map from original output value name to merged output value.
        merged_outputs = defaultdict(Value)

        # Partition the input values according to the number of microbatches.
        self._partition_inputs(module, transformed_module, pipelined_value_map)

        # Schedule ops on each device in order of increasing timestep.
        for timestep in range(len(self._schedule)):
            for device in self._schedule[timestep]:
                (op_name, microbatch) = self._schedule[timestep][device]
                orig_op = module.get_op(op_name)
                orig_inputs = orig_op.get_in_edges()
                orig_outputs = orig_op.get_out_edges()

                # Collect the pipelined input values for this op.
                pipelined_inputs = []
                for orig_input in orig_inputs:
                    pipelined_input_map = pipelined_value_map[orig_input.name]
                    pipelined_input = pipelined_input_map[microbatch]
                    # Send the input value to the correct device, if necessary.
                    if pipelined_input.type.device != device:
                        pipelined_input_map[microbatch] = transformed_module.add_op(
                            "Send",
                            f"Send/{pipelined_input.name}",
                            attributes={"device": device},
                            inputs=[pipelined_input],
                            output_names=[f"{pipelined_input.name}@{device}"],
                        )
                        # If the input value we are sending is a module input and it is
                        # not being partitioned, replicate the output of the send
                        # for all other microbatches.
                        if (
                            module.is_input(orig_input.name)
                            and not orig_input.name in self._batch_dims
                        ):
                            for mb in pipelined_input_map:
                                if mb != microbatch:
                                    pipelined_input_map[mb] = pipelined_input_map[
                                        microbatch
                                    ]
                    pipelined_inputs.append(pipelined_input_map[microbatch])

                # Add the pipelined version of the op for the given microbatch to
                # the transformed module.
                pipelined_output_names = [
                    f"{orig_output.name}_{microbatch}" for orig_output in orig_outputs
                ]
                pipelined_outputs = transformed_module.add_op(
                    orig_op.op_type,
                    name=f"{orig_op.name}_{microbatch}",
                    attributes=orig_op._attributes,
                    inputs=pipelined_inputs,
                    output_names=pipelined_output_names,
                )

                # Update the pipelined value map with the newly generated output values.
                if not isinstance(pipelined_outputs, tuple):
                    pipelined_outputs = (pipelined_outputs,)
                for (orig_output, pipelined_output) in zip(
                    orig_outputs, pipelined_outputs
                ):
                    pipelined_value_map[orig_output.name][microbatch] = pipelined_output

                    # Aggregate outputs.
                    if module.is_output(orig_output.name):
                        num_completed_microbatches = len(
                            pipelined_value_map[orig_output.name]
                        )
                        self._aggregate_outputs(
                            transformed_module,
                            orig_output,
                            pipelined_output,
                            merged_outputs,
                            num_completed_microbatches,
                        )

        return transformed_module

    # TODO: Move this to a separate scheduler
    """
    # Enumerate the ops to schedule on each device across all microbatches.
    ops_to_schedule = defaultdict(lambda: defaultdict(set))
    for i in range(self._num_microbatches):
        for op_name in module.get_ops():
            device = partition_map[op_name]
            ops_to_schedule[device].add((op_name, i))

    # Schedule the ops.
    # TODO: Move this to a modular scheduler?
    schedule = defaultdict(lambda: defaultdict(list))
    consumers = 
    timestep = 0
    done = False
    while not done:
        for device in ops_to_schedule: 
            ready_ops = []  # TODO: Get ready ops
            if len(ready_ops) > 0:
                schedule[timestep][device] = ready_ops[0]
                ops_to_schedule[device].remove(ready_ops[0])
        timestemp += 1
        done = any([len(ops_to_schedule[device]) > 0 for device in ops_to_schedule])
    """
