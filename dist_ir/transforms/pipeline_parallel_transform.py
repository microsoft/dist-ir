from ..ir.module import Module

import copy


class PipelineParallelTransform:
    """Partitions a module using pipeline parallelism.

    Attributes:
      num_microbatches: The number of microbatches per pipeline iteration.
      partition_map: A map from input value name to partition dimension.
      device_map: A map from op name to device. (TODO: better name for this?)
    """

    def __init__(self, num_microbatches, partition_map, device_map):
        self._num_microbatches = num_microbatches
        self._partition_map = partition_map
        self._device_map = device_map

    def apply(self, module):

        transformed_module = Module()

        # Partition the input values according to the number of microbatches.
        pipelined_input_values = {}
        input_values = module.get_inputs()
        for input_value in input_values:
            v = transformed_module.add_input_value(
                input_value.name, copy.deepcopy(input_value.type)
            )
            if input_value.name in self._partition_map:
                # TODO: Return an actual tuple of values instead of a TupleType?
                vs = transformed_module.add_op(
                    "Split",
                    name=f"Split/{v.name}",
                    inputs=[v],
                    attributes={
                        "num_splits": self._num_microbatches,
                        "split_dim": self._partition_map[input_value.name],
                    },
                    output_names=[f"{v.name}s"],
                )
                pipelined_input_values[input_value.name] = vs

        # TODO: Should we explicitly unroll every microbatch or use a specialized op
        #       (similar to Pmap)?
        # TODO: Should we explicitly tie parallel ops together with Par primitive?
        """
        Initialize a map from original value to pipelined value, for example:
            {
                x: {
                    1: x_1
                    2: x_2
                    ...
                    k: x_k
                }, 
                z: {
                    1: z_1,
                    2: z_2,
                    ...
                    k: z_k
                },
            }
        For each microbatch i:
            For each op in the original module:
                - Lookup the pipelined input values in the value map
                - Move the input values to the appropriate devices if necessary
                  (as determined by the device map)
                - Add the op with the pipelined input values
                - Add the op outputs to the value map
            Aggregate the outputs for the microbatch
        """
