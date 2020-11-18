from ..ir.module import Module


class DataParallelTransform:
    """Partitions a module using data parallelism.

    Replicates the given model across devices by instantiating an identical version
    of the model on each device. The user specifies which input values to
    partition between each device as well as the dimension to partition for each input
    (e.g. selecting the first dimension for the input minibatch would partition
    along the batch dimension). The selected input values are scattered between
    each device, while the remaining input values are broadcasted. The module will
    be replicated using a Pmap operator. The original output values are retrieved
    from each replica through Allreduce operators.

    Attributes:
      partition_map: A map from Value name to partition dimension.
      num_partitions: The number of partitions to split the model into.
    """

    def __init__(self, partition_map, num_partitions):
        self._partition_map = partition_map
        self._num_partitions = num_partitions

    def apply(self, module):
        """Applies the transformation to the given module and returns the transformed module."""
        transformed_module = Module()

        # Initialize a map for keeping track of which partitioned values on each device
        # correspond to values in the submodule.
        value_name_map = {}
        devices = list(range(self._num_partitions))
        for device in devices:
            value_name_map[device] = {}
        pmap_inputs = []

        # Either scatter or broadcast each input value depending on what the user
        # has requested.
        input_values = module.get_inputs()
        for input_value in input_values:
            if input_value.name in self._partition_map:
                v = transformed_module.add_input_value(
                    input_value.name, input_value.type
                )
                scattered_v = transformed_module.add_op(
                    "Scatter",
                    name=f"Scatter/{v.name}",
                    inputs=[v],
                    attributes={
                        "devices": list(range(self._num_partitions)),
                        "num_splits": self._num_partitions,
                        "split_dim": self._partition_map[input_value.name],
                    },
                    output_names=[f"{v.name}_{i}" for i in range(self._num_partitions)],
                )
                for device in devices:
                    value_name_map[device][input_value.name] = scattered_v[device].name
                    pmap_inputs.append(scattered_v[device])
            else:
                v = transformed_module.add_input_value(
                    input_value.name, input_value.type
                )
                broadcasted_v = transformed_module.add_op(
                    "Broadcast",
                    name=f"Broadcast/{v.name}",
                    inputs=[v],
                    attributes={"devices": devices},
                    output_names=[f"{v.name}_{i}" for i in range(self._num_partitions)],
                )
                for device in devices:
                    value_name_map[device][input_value.name] = broadcasted_v[
                        device
                    ].name
                    pmap_inputs.append(broadcasted_v[device])

        # Add the Pmap operator to the transformed module. The Pmap operator will
        # encapsulate the original module.
        output_values = module.get_outputs()
        pmap_output_names = []
        for device in devices:
            for i, output_value in enumerate(output_values):
                pmap_output_name = f"{output_value.name}_{device}"
                value_name_map[device][output_value.name] = pmap_output_name
                pmap_output_names.append(pmap_output_name)
        partitioned_output_values = transformed_module.add_op(
            "Pmap",
            inputs=pmap_inputs,
            attributes={"devices": devices},
            metadata={"value_name_map": value_name_map},
            submodules=[module],
            output_names=pmap_output_names,
        )

        # Add Allreduce operators to collect output values from each device.
        for j, output_value in enumerate(output_values):
            allreduce_inputs = []
            for i, device in enumerate(devices):
                allreduce_inputs.append(
                    partitioned_output_values[i * len(output_values) + j]
                )
            transformed_module.add_op(
                "Allreduce",
                name=f"Allreduce/{output_value.name}",
                inputs=allreduce_inputs,
                output_names=[output_value.name],
            )

        return transformed_module
