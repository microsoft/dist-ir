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
      devices: The devices over which to partition the model.
    """

    def __init__(self, partition_map, devices):
        self._partition_map = partition_map
        self._devices = devices

    def apply(self, module):
        """Applies the transformation to the given module and returns the transformed module."""
        transformed_module = Module()

        # Initialize a map for keeping track of which partitioned values on each device
        # correspond with the input and output values of the original module.
        value_name_map = {}
        for device in self._devices:
            value_name_map[device] = {}
        pmap_inputs = []

        # Either scatter or broadcast each input value depending on what the user
        # has requested.
        input_values = module.get_inputs()
        for input_value in input_values:
            if input_value.name in self._partition_map:
                v = transformed_module.add_input_value(
                    input_value.name,
                    input_value.type,
                    input_value.device,
                )
                scattered_v = transformed_module.add_op(
                    "Scatter",
                    name=f"Scatter/{v.name}",
                    inputs=[v],
                    device=v.device,
                    attributes={
                        "devices": self._devices,
                        "split_dim": self._partition_map[input_value.name],
                    },
                    output_names=[
                        f"{v.name}_{device.device_id}" for device in self._devices
                    ],
                )
                for i, device in enumerate(self._devices):
                    value_name_map[device][input_value.name] = scattered_v[i].name
                    pmap_inputs.append(scattered_v[i])
            else:
                v = transformed_module.add_input_value(
                    input_value.name, input_value.type
                )
                broadcasted_v = transformed_module.add_op(
                    "Broadcast",
                    name=f"Broadcast/{v.name}",
                    inputs=[v],
                    device=v.device,
                    attributes={"devices": self._devices},
                    output_names=[
                        f"{v.name}_{device.device_id}" for device in self._devices
                    ],
                )
                for i, device in enumerate(self._devices):
                    value_name_map[device][input_value.name] = broadcasted_v[i].name
                    pmap_inputs.append(broadcasted_v[i])

        # Add the Pmap operator to the transformed module. The Pmap operator will
        # encapsulate the original module.
        output_values = module.get_outputs()
        pmap_output_names = []
        for device in self._devices:
            for i, output_value in enumerate(output_values):
                pmap_output_name = f"{output_value.name}_{device.device_id}"
                value_name_map[device][output_value.name] = pmap_output_name
                pmap_output_names.append(pmap_output_name)
        partitioned_output_values = transformed_module.add_op(
            "Pmap",
            inputs=pmap_inputs,
            attributes={"devices": self._devices},
            metadata={"value_name_map": value_name_map},
            submodules=[module],
            output_names=pmap_output_names,
        )

        # Add Allreduce operators to collect output values from each device.
        for j, output_value in enumerate(output_values):
            allreduce_inputs = []
            for i, device in enumerate(self._devices):
                allreduce_inputs.append(
                    partitioned_output_values[i * len(output_values) + j]
                )
            transformed_module.add_op(
                "Allreduce",
                name=f"Allreduce/{output_value.name}",
                inputs=allreduce_inputs,
                output_names=[output_value.name],
                device=output_value.device,
            )

        return transformed_module
