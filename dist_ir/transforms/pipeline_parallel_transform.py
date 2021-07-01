from collections import defaultdict

from ..ir.function import FunctionMaker
from ..ir.value import Value
from . import utils


class PipelineParallelTransform:
    """Partitions a function using pipeline parallelism.

    Attributes:
      num_microbatches: The number of microbatches per pipeline iteration.
      batch_dims: A map from input value to partition dimension.
      reduction_ops: A map from output value to a map of reduction op params.
      partition_map: A map from op to device.
      schedule: A list of maps from device to a tuple of (op, microbatch).
    """

    def __init__(
        self, num_microbatches, batch_dims, reduction_params, partition_map, schedule
    ):
        self._num_microbatches = num_microbatches
        self._batch_dims = batch_dims
        self._reduction_params = reduction_params
        self._partition_map = partition_map
        self._schedule = schedule
        self._op_to_stage_map = utils.get_op_to_stage_map(
            list(self._partition_map.keys())
        )

    def _forward_value(self, transformed_function, value, device):
        """Forwards the specified value to the specified device by adding a Send op."""
        forwarded_value = transformed_function.add_op(
            "Send",
            name=f"Send/{value.name}@{device}",
            inputs=[value],
            attributes={"device": device},
            output_names=[f"{value.name}@{device}"],
        )
        return forwarded_value

    def _partition_inputs(self, function, transformed_function, pipelined_value_map):
        """Splits the input values according to the number of specified microbatches."""
        for input_value in function.inputs:
            v = transformed_function.add_input_value(input_value.name, input_value.type)
            pipelined_input_map = pipelined_value_map[input_value]
            if input_value in self._batch_dims:
                vs = transformed_function.add_op(
                    "SplitDistIR",
                    name=f"Split/{v.name}",
                    inputs=[v],
                    attributes={
                        "num_splits": self._num_microbatches,
                        "axis": self._batch_dims[input_value],
                    },
                    output_names=[f"{v.name}s"],
                )
                for i in range(self._num_microbatches):
                    v_i = transformed_function.add_op(
                        "Select",
                        name=f"Select/{v.name}_{i}",
                        attributes={"index": i},
                        inputs=[vs],
                        output_names=[f"{v.name}_{i}"],
                    )
                    pipelined_input_map[i] = v_i
            else:
                for i in range(self._num_microbatches):
                    pipelined_input_map[i] = v

            # Forward the input value(s) if the destination device(s) are not
            # the same as the source device.
            input_device = input_value.type.device
            consumer_ops = function.consumers[input_value]
            consumer_stages = utils.get_stages_from_ops(
                self._op_to_stage_map, consumer_ops
            )
            consumer_devices = set([self._partition_map[c] for c in consumer_stages])
            for consumer_device in consumer_devices:
                if consumer_device != input_device:
                    # For the first microbatch we always forward the value. For subsequent
                    # microbatches, we only forward the value if the user requested this input
                    # to be partitioned (e.g. the labels value might be partitioned and then
                    # each partition will be forwarded to the device running the loss stage).
                    # For other input values (e.g. weight tensors), we only need to forward the
                    # value once because the value will be identical across microbatches.
                    #
                    # TODO: Propagate these values alongside activations instead of sending them
                    # ahead of time to be consistent with ORT?
                    for i in range(self._num_microbatches):
                        if input_value in self._batch_dims or i == 0:
                            forwarded_input = self._forward_value(
                                transformed_function,
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
        transformed_function,
        orig_output,
        pipelined_output,
        merged_output_map,
        num_completed_microbatches,
    ):
        """Aggregates the specified output according to the user-provided reduction parameters.

        Args:
          transformed_function: The transformed function.
          orig_output: The original version of the output value.
          pipelined_output: The transformed (i.e. partitioned) version of the output value.
          merged_output_map: A map from original output value to aggregated output value.
          num_completed_microbatches: The number of microbatches completed so far.
        """
        if self._reduction_params[orig_output] is None:
            # This output does not need to be aggregated.
            return

        reduction_op_type = self._reduction_params[orig_output]["op_type"]
        if num_completed_microbatches == 1:
            merged_output_map[orig_output] = pipelined_output
        else:
            merged_output = merged_output_map[orig_output]

            # Prepare the reduction op name and output value name.
            op_name = (
                f"{reduction_op_type}/{merged_output.name}-{pipelined_output.name}"
            )
            if num_completed_microbatches == self._num_microbatches:
                output_name = orig_output.name
            else:
                output_name = f"{orig_output.name}/merged_{num_completed_microbatches}"

            # Add the requested reduction op to the transformed function.
            if reduction_op_type == "Add":
                merged_output_map[orig_output] = transformed_function.add_op(
                    "Add",
                    name=op_name,
                    inputs=[merged_output, pipelined_output],
                    output_names=[output_name],
                )
            elif reduction_op_type == "Concat":
                dim = self._reduction_params[orig_output]["dim"]
                merged_output_map[orig_output] = transformed_function.add_op(
                    "Concat",
                    attributes={"axis": dim},
                    name=op_name,
                    inputs=[merged_output, pipelined_output],
                    output_names=[output_name],
                )
            else:
                raise ValueError(
                    f"Unknown reduction op type {reduction_op_type} "
                    f"for output {orig_output}"
                )

    def apply(self, function):
        """Applies the transformation to the function and returns a transformed function."""

        transformed_function = FunctionMaker()

        # A map from original value to another map from microbatch number to
        # pipelined value.
        pipelined_value_map = defaultdict(lambda: {})

        # A map from original output value to merged output value.
        merged_output_map = defaultdict(Value)

        # Partition the input values according to the number of microbatches.
        self._partition_inputs(function, transformed_function, pipelined_value_map)

        # Schedule stages on each device in order of increasing timestep.
        for timestep in range(len(self._schedule)):
            for device in self._schedule[timestep]:
                # Look up the next stage to execute according to the schedule
                # and add each op in the stage to the transformed function.
                (stage, microbatch) = self._schedule[timestep][device]
                for orig_op in stage.ops:
                    orig_inputs = orig_op.inputs
                    orig_outputs = orig_op.outputs

                    # Collect the pipelined input values for this op.
                    pipelined_inputs = []
                    for orig_input in orig_inputs:
                        pipelined_input_map = pipelined_value_map[orig_input]
                        pipelined_inputs.append(pipelined_input_map[microbatch])

                    # Add the pipelined version of the op for the given microbatch to
                    # the transformed function.
                    pipelined_output_names = [
                        f"{orig_output.name}_{microbatch}"
                        for orig_output in orig_outputs
                    ]
                    pipelined_outputs = transformed_function.add_op(
                        orig_op.op_type,
                        name=f"{orig_op.name}_{microbatch}",
                        attributes=orig_op.attributes,
                        inputs=pipelined_inputs,
                        output_names=pipelined_output_names,
                    )

                    # Update the pipelined value map with the newly generated
                    # output values.
                    if not isinstance(pipelined_outputs, tuple):
                        pipelined_outputs = (pipelined_outputs,)
                    for (orig_output, pipelined_output) in zip(
                        orig_outputs, pipelined_outputs
                    ):
                        pipelined_output_map = pipelined_value_map[orig_output]
                        pipelined_output_map[microbatch] = pipelined_output

                        if orig_output not in stage.outputs:
                            # This output is an intermediate output *within* a stage which does not
                            # require any additional processing.
                            continue
                        elif orig_output in function.outputs:
                            # This output is a function output, which means we need to aggregate it
                            # with all other corresponding partitioned outputs for each microbatch.
                            num_completed_microbatches = len(pipelined_output_map)
                            self._aggregate_outputs(
                                transformed_function,
                                orig_output,
                                pipelined_output,
                                merged_output_map,
                                num_completed_microbatches,
                            )
                        else:
                            # This output is an intermediate stage output, which means we need to
                            # forward the output to the next stage if the next stage is located on
                            # a different device.
                            consumer_ops = function.consumers[orig_output]
                            consumer_stages = utils.get_stages_from_ops(
                                self._op_to_stage_map, consumer_ops
                            )
                            consumer_devices = {
                                self._partition_map[c] for c in consumer_stages
                            }
                            for consumer_device in consumer_devices:
                                if device != consumer_device:
                                    pipelined_output_map[
                                        microbatch
                                    ] = self._forward_value(
                                        transformed_function,
                                        pipelined_output,
                                        consumer_device,
                                    )

        return transformed_function.finalize()
