from ..ir.function import FunctionMaker

import copy


class HorizontalParallelTransform:
    """Partitions a function using horizontal parallelism.

    Attributes:
      ops: The ops affected by the transform.
      batch_dims: A map from input value name to partition dimension.
      reduction_params: A map from output value name to a map of reduction op params.
      devices: The devices over which to partition the model.
    """

    def __init__(self, ops, param_dims, reduction_params, devices):
        self._ops = ops
        self._param_dims = param_dims
        self._reduction_params = reduction_params
        self._devices = devices

    def apply(self, function, verify_fn=None):
        """Applies the transformation to the given function and returns the transformed function."""
        transformed_function = FunctionMaker()
        subfunction = function.get_subfunction(self._ops)
        if verify_fn is not None:
            if not verify_fn(subfunction):
                return None

        # Either scatter or broadcast each input value depending on what the user
        # has requested.
        # TODO: Add explicit Send ops if the source device is not one oGf the
        #       destination devices.
        value_map = {}

        for input_value in function.inputs:
            value_map[input_value] = transformed_function.add_input_value(
                input_value.name, input_value.type
            )

        added_pmap = False
        for op in function.ops:
            if op in subfunction.ops:
                # Only add the Pmap op the first time we encounter a subfunction op.
                # Wait for all subfunction inputs to be added to the transformed module before
                # adding the Pmap op.
                if added_pmap or not all(v in value_map for v in subfunction.inputs):
                    continue

                pmap_input_values = []
                for input_value in subfunction.inputs:
                    v = value_map[input_value]
                    if input_value in self._param_dims:
                        vs = transformed_function.add_op(
                            "Scatter",
                            name=f"Scatter/{v.name}",
                            inputs=[v],
                            attributes={
                                "devices": self._devices,
                                "dim": self._param_dims[input_value],
                            },
                            output_names=[f"{v.name}s"],
                        )
                    else:
                        vs = transformed_function.add_op(
                            "Broadcast",
                            name=f"Broadcast/{v.name}",
                            inputs=[v],
                            attributes={"devices": self._devices},
                            output_names=[f"{v.name}s"],
                        )
                    pmap_input_values.append(vs)

                # Add the Pmap operator to the transformed function. The Pmap operator will
                # encapsulate the original function.
                pmap_output_names = []
                for i, output_value in enumerate(subfunction.outputs):
                    pmap_output_name = f"{output_value.name}is"
                    pmap_output_names.append(pmap_output_name)
                pmap_output_values = transformed_function.add_op(
                    "Pmap",
                    inputs=pmap_input_values,
                    attributes={"devices": self._devices},
                    subfunctions=[subfunction],
                    output_names=pmap_output_names,
                )

                if not isinstance(pmap_output_values, tuple):
                    pmap_output_values = (pmap_output_values,)

                # Add reduction operators to collect output values from each device.
                # TODO: Add explicit Send ops if the destination device is not one of the
                #       source devices.
                for i, output_value in enumerate(subfunction.outputs):
                    reduction_op_type = self._reduction_params[output_value]["op_type"]
                    if reduction_op_type == "Allreduce":
                        pmap_output = transformed_function.add_op(
                            "Allreduce",
                            name=f"Allreduce/{output_value.name}",
                            inputs=[pmap_output_values[i]],
                            output_names=[f"{output_value.name}s"],
                        )
                    elif reduction_op_type == "Gather":
                        dim = self._reduction_params[output_value]["dim"]
                        device = self._reduction_params[output_value]["device"]
                        pmap_output = transformed_function.add_op(
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
                    value_map[output_value] = pmap_output
                added_pmap = True
            else:
                inputs = [value_map[v] for v in op.inputs]
                output_names = [v.name for v in op.outputs]

                outputs = transformed_function.add_op(
                    op.op_type,
                    name=op.name,
                    inputs=inputs,
                    attributes=op.attributes,
                    subfunctions=op.subfunctions,
                    output_names=output_names,
                )
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                for output in outputs:
                    value_map[output] = output
        return transformed_function.finalize()
