from ..ir.module import Module

import copy


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
      batch_dims: A map from input value name to partition dimension.
      reduction_params: A map from output value name to a map of reduction op params.
      devices: The devices over which to partition the model.
    """

    def __init__(self, batch_dims, reduction_params, devices):
        self._batch_dims = batch_dims
        self._reduction_params = reduction_params
        self._devices = devices

    def apply(self, module):
        """Applies the transformation to the given module and returns the transformed module."""
        transformed_module = Module()

        # Either scatter or broadcast each input value depending on what the user
        # has requested.
        # TODO: Add explicit Send ops if the source device is not one of the
        #       destination devices.
        input_values = module.get_inputs()
        pmap_input_values = []
        for input_value in input_values:
            v = transformed_module.add_input_value(
                input_value.name, copy.deepcopy(input_value.type)
            )
            if input_value.name in self._batch_dims:
                vs = transformed_module.add_op(
                    "Scatter",
                    name=f"Scatter/{v.name}",
                    inputs=[v],
                    attributes={
                        "devices": self._devices,
                        "dim": self._batch_dims[input_value.name],
                    },
                    output_names=[f"{v.name}s"],
                )
            else:
                vs = transformed_module.add_op(
                    "Broadcast",
                    name=f"Broadcast/{v.name}",
                    inputs=[v],
                    attributes={"devices": self._devices},
                    output_names=[f"{v.name}s"],
                )
            pmap_input_values.append(vs)

        # Add the Pmap operator to the transformed module. The Pmap operator will
        # encapsulate the original module.
        output_values = module.get_outputs()
        pmap_output_names = []
        for i, output_value in enumerate(output_values):
            pmap_output_name = f"{output_value.name}is"
            pmap_output_names.append(pmap_output_name)
        pmap_output_values = transformed_module.add_op(
            "Pmap",
            inputs=pmap_input_values,
            attributes={"devices": self._devices},
            submodules=[module],
            output_names=pmap_output_names,
        )

        if not isinstance(pmap_output_values, tuple):
            pmap_output_values = (pmap_output_values,)

        # Add reduction operators to collect output values from each device.
        # TODO: Add explicit Send ops if the destination device is not one of the
        #       source devices.
        for i, output_value in enumerate(output_values):
            reduction_op_type = self._reduction_params[output_value.name]["op_type"]
            if reduction_op_type == "Allreduce":
                transformed_module.add_op(
                    "Allreduce",
                    name=f"Allreduce/{output_value.name}",
                    inputs=[pmap_output_values[i]],
                    output_names=[f"{output_value.name}s"],
                )
            elif reduction_op_type == "Gather":
                dim = self._reduction_params[output_value.name]["dim"]
                device = self._reduction_params[output_value.name]["device"]
                transformed_module.add_op(
                    "Gather",
                    name=f"Gather/{output_value.name}",
                    inputs=[pmap_output_values[i]],
                    attributes={"dim": dim, "device": device},
                    output_names=[f"{output_value.name}s"],
                )
            else:
                raise ValueError(
                    f"Unknown reduction op type {reduction_op_type} for "
                    f"output value {output_value}"
                )

        return transformed_module
