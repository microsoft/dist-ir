from ..ir.module import Module

import copy


class HorizontalParallelTransform:
    """Partitions a module using horizontal parallelism.

    Attributes:
      op_names: The names of ops captured by the transform.
      batch_dims: A map from input value name to partition dimension.
      reduction_params: A map from output value name to a map of reduction op params.
      devices: The devices over which to partition the model.
    """

    def __init__(self, op_names, param_dims, reduction_params, devices):
        self._op_names = op_names
        self._param_dims = param_dims
        self._reduction_params = reduction_params
        self._devices = devices

    def apply(self, module):
        """Applies the transformation to the given module and returns the transformed module."""
        transformed_module = Module()
        submodule = module.get_submodule(self._op_names)

        # Either scatter or broadcast each input value depending on what the user
        # has requested.
        # TODO: Add explicit Send ops if the source device is not one oGf the
        #       destination devices.
        input_values = module.get_inputs()
        pmap_input_values = []
        value_map = {}
        for input_value in input_values:
            # TODO: Remove deepcopy of type?
            v = transformed_module.add_input_value(
                input_value.name, copy.deepcopy(input_value.type)
            )
            value_map[v.name] = v
            if input_value.name in self._param_dims:
                vs = transformed_module.add_op(
                    "Scatter",
                    name=f"Scatter/{v.name}",
                    inputs=[v],
                    attributes={
                        "devices": self._devices,
                        "dim": self._param_dims[input_value.name],
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

        added_pmap = False
        for op_name, op in module.get_ops().items():
            if submodule.is_op(op_name):
                # Only add the Pmap op the first time we encounter a submodule op.
                if added_pmap:
                    continue

                # Add the Pmap operator to the transformed module. The Pmap operator will
                # encapsulate the original module.
                output_values = submodule.get_outputs()
                pmap_output_names = []
                for i, output_value in enumerate(output_values):
                    pmap_output_name = f"{output_value.name}is"
                    pmap_output_names.append(pmap_output_name)
                pmap_output_values = transformed_module.add_op(
                    "Pmap",
                    inputs=pmap_input_values,
                    attributes={"devices": self._devices},
                    submodules=[submodule],
                    output_names=pmap_output_names,
                )

                if not isinstance(pmap_output_values, tuple):
                    pmap_output_values = (pmap_output_values,)

                # Add reduction operators to collect output values from each device.
                # TODO: Add explicit Send ops if the destination device is not one of the
                #       source devices.
                for i, output_value in enumerate(output_values):
                    reduction_op_type = self._reduction_params[output_value.name][
                        "op_type"
                    ]
                    if reduction_op_type == "Allreduce":
                        pmap_output = transformed_module.add_op(
                            "Allreduce",
                            name=f"Allreduce/{output_value.name}",
                            inputs=[pmap_output_values[i]],
                            output_names=[f"{output_value.name}s"],
                        )
                    elif reduction_op_type == "Gather":
                        dim = self._reduction_params[output_value.name]["dim"]
                        device = self._reduction_params[output_value.name]["device"]
                        pmap_output = transformed_module.add_op(
                            "Gather",
                            name=f"Gather/{output_value.name}",
                            inputs=[pmap_output_values[i]],
                            attributes={"dim": dim, "device": device},
                            output_names=[f"{output_value.name}"],
                        )
                    else:
                        raise ValueError(
                            f"Unknown reduction op type {reduction_op_type} for "
                            f"output value {output_value}"
                        )
                    value_map[output_value.name] = pmap_output
                added_pmap = True
            else:
                inputs = []
                for input in op.get_in_edges():
                    inputs.append(value_map[input.name])
                output_names = [output.name for output in op.get_out_edges()]

                outputs = transformed_module.add_op(
                    op.op_type,
                    name=op.name,
                    inputs=inputs,
                    attributes=copy.deepcopy(op._attributes),
                    submodules=copy.deepcopy(op._submodules),
                    output_names=output_names,
                )
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                for output in outputs:
                    value_map[output.name] = output
        transformed_module.finalize()
        return transformed_module
