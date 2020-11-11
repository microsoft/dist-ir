from ..ir.module import Module


class DataParallelTransform:
    def __init__(self, partitioned_input_name, partition_dim, num_partitions):
        self._partitioned_input_name = partitioned_input_name
        self._partition_dim = partition_dim
        self._num_partitions = num_partitions

    def apply(self, module):
        transformed_module = Module()

        partitioned_values = {}
        for input_value in module.inputs:
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
                partitioned_values[input_value.name] = scattered_v
            else:
                v = transformed_module.add_input_value(
                    input_value.name, input_value.type
                )
                broadcasted_v = transformed_module.add_op(
                    "Broadcast",
                    name=f"Broadcast/{v.name}",
                    inputs=[v],
                    attributes={"devices": list(range(self._num_partitions))},
                    output_names=[f"{v.name}_{i}" for i in range(self._num_partitions)],
                )
                partitioned_values[input_value.name] = broadcasted_v

        ops = module.get_ops()
        for i in range(self._num_partitions):
            for op_name, op in ops.items():
                inputs = []
                output_names = []
                for in_edge in op.get_in_edges():
                    inputs.append(partitioned_values[in_edge.name][i])
                for out_edge in op.get_out_edges():
                    partitioned_values[out_edge.name] = []
                    output_names.append(f"{out_edge.name}_{i}")
                # TODO: Handle attributes and submodules
                outputs = transformed_module.add_op(
                    op.op_type,
                    name=f"{op_name}_{i}",
                    inputs=inputs,
                    output_names=output_names,
                )
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                for j, out_edge in enumerate(op.get_out_edges()):
                    partitioned_values[out_edge.name].append(outputs[j])

        return transformed_module
