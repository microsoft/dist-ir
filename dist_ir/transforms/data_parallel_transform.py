from ..ir.module import Module


class DataParallelTransform:
    def __init__(self, partitioned_input_name, partition_dim, num_partitions):
        self._partitioned_input_name = partitioned_input_name
        self._partition_dim = partition_dim
        self._num_partitions = num_partitions

    def apply(self, module):
        transformed_module = Module()

        value_map = {}
        devices = list(range(self._num_partitions))
        for device in devices:
            value_map[device] = {}
        pmap_inputs = []

        input_values = module.get_inputs()
        for input_value in input_values:
            if input_value.name == self._partitioned_input_name:
                v = transformed_module.add_input_value(
                    input_value.name, input_value.type
                )
                scattered_v = transformed_module.add_op(
                    "Scatter",
                    name=f"Scatter/{v.name}",
                    inputs=[v],
                    attributes={
                        "devices": list(range(self._num_partitions)),
                        "split_dim": self._partition_dim,
                    },
                    output_names=[f"{v.name}_{i}" for i in range(self._num_partitions)],
                )
                for device in devices:
                    value_map[device][input_value.name] = scattered_v[device]
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
                    value_map[device][input_value.name] = broadcasted_v[device]
                    pmap_inputs.append(broadcasted_v[device])

        output_values = module.get_outputs()
        pmap_output_names = []
        for device in devices:
            for output_value in output_values:
                pmap_output_names.append(f"{output_value.name}_{device}")

        partitioned_output_values = transformed_module.add_op(
            "Pmap",
            inputs=pmap_inputs,
            attributes={"devices": devices},
            metadata={"value_map": value_map},
            submodules=[module],
            output_names=pmap_output_names,
        )

        output_map = {}
        num_output_values = len(output_values)
        for device in devices:
            output_map[device] = {}
            for i, output_value in enumerate(output_values):
                output_map[device][output_value.name] = partitioned_output_values[
                    device * num_output_values + i
                ]

        for output_value in output_values:
            allreduce_inputs = [
                output_map[device][output_value.name] for device in devices
            ]
            transformed_module.add_op(
                "Allreduce",
                inputs=allreduce_inputs,
                output_names=[output_value.name],
            )

        return transformed_module
