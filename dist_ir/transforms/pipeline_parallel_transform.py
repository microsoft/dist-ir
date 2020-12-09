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
      partition_map: A map from op name to device.
      schedule: A list of maps from device to a tuple of (op_name, microbatch).
    """

    def __init__(
        self, num_microbatches, batch_dims, reduction_params, partition_map, schedule
    ):
        self._num_microbatches = num_microbatches
        self._batch_dims = batch_dims
        self._reduction_params = reduction_params
        self._partition_map = partition_map
        self._schedule = schedule

    def _forward_value(self, transformed_module, value, device):
        forwarded_value = transformed_module.add_op(
            "Send",
            name=f"Send/{value.name}@{device}",
            inputs=[value],
            attributes={"device": device},
            output_names=[f"{value.name}@{device}"],
        )
        return forwarded_value

    def _partition_inputs(self, module, transformed_module, pipelined_value_map):
        input_values = module.get_inputs()
        for input_value in input_values:
            v = transformed_module.add_input_value(
                input_value.name, copy.deepcopy(input_value.type)
            )
            pipelined_input_map = pipelined_value_map[input_value.name]
            if input_value.name in self._batch_dims:
                vs = transformed_module.add_op(
                    "Split",
                    name=f"Split/{v.name}",
                    inputs=[v],
                    attributes={
                        "num_splits": self._num_microbatches,
                        "dim": self._batch_dims[input_value.name],
                    },
                    output_names=[f"{v.name}s"],
                )
                for i in range(self._num_microbatches):
                    v_i = transformed_module.add_op(
                        "Select",
                        name=f"Select/{v.name}_{i}",
                        attributes={"dim": i},
                        inputs=[vs],
                        output_names=[f"{v.name}_{i}"],
                    )
                    pipelined_input_map[i] = v_i
            else:
                for i in range(self._num_microbatches):
                    pipelined_input_map[i] = v

            # Forward the input value(s) if necessary.
            input_device = input_value.type.device
            consumers = module.get_consumers_for_value(input_value.name)
            consumer_devices = set([self._partition_map[c] for c in consumers])
            for consumer_device in consumer_devices:
                if consumer_device != input_device:
                    for i in range(self._num_microbatches):
                        if input_value.name in self._batch_dims or i == 0:
                            forwarded_input = self._forward_value(
                                transformed_module,
                                pipelined_input_map[i],
                                consumer_device,
                            )
                        else:
                            # If this is not a batch-dependent value then there only
                            # needs to be one forwarded copy for all microbatches.
                            forwarded_input = pipelined_input_map[0]
                        pipelined_input_map[i] = forwarded_input

    def _aggregate_outputs(
        self,
        transformed_module,
        orig_output,
        pipelined_output,
        merged_outputs,
        num_completed_microbatches,
    ):
        if self._reduction_params[orig_output.name] is None:
            # This output does not need to be aggregated.
            return

        reduction_op_type = self._reduction_params[orig_output.name]["op_type"]
        if num_completed_microbatches == 1:
            merged_outputs[orig_output.name] = pipelined_output
        else:
            merged_output = merged_outputs[orig_output.name]

            # Forward the output value if necessary.
            if merged_output.type.device != pipelined_output.type.device:
                pipelined_output = self._forward_value(
                    transformed_module, pipelined_output, merged_output.type.device
                )

            # Prepare the reduction op name and output value name.
            op_name = (
                f"{reduction_op_type}/{merged_output.name}-{pipelined_output.name}"
            )
            if num_completed_microbatches == self._num_microbatches:
                output_name = orig_output.name
            else:
                output_name = f"{orig_output.name}/merged_{num_completed_microbatches}"

            # Add the requested reduction op to the transformed module.
            if reduction_op_type == "Add":
                merged_outputs[orig_output.name] = transformed_module.add_op(
                    "Add",
                    name=op_name,
                    inputs=[merged_output, pipelined_output],
                    output_names=[output_name],
                )
            elif reduction_op_type == "Concat":
                dim = self._reduction_params[orig_output.name]["dim"]
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

        # Partition the input values for each microbatch.
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
                    pipelined_output_map = pipelined_value_map[orig_output.name]
                    pipelined_output_map[microbatch] = pipelined_output

                    # Aggregate outputs.
                    if module.is_output(orig_output.name):
                        num_completed_microbatches = len(pipelined_output_map)
                        self._aggregate_outputs(
                            transformed_module,
                            orig_output,
                            pipelined_output,
                            merged_outputs,
                            num_completed_microbatches,
                        )
                    else:
                        # Forward the output value, if necessary.
                        consumers = module.get_consumers_for_value(orig_output.name)
                        consumer_devices = set(
                            [self._partition_map[c] for c in consumers]
                        )
                        for consumer_device in consumer_devices:
                            if device != consumer_device:
                                pipelined_output_map[microbatch] = self._forward_value(
                                    transformed_module,
                                    pipelined_output,
                                    consumer_device,
                                )

        return transformed_module
