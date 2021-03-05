from ..ir.function import FunctionMaker
from ..ir import Device

import copy


def shard_transform(
    function, ops, input_dims, reduction_params, devices, verify_fn=None
):
    """Maps a function across devices using a parallel map (pmap) operator.

    This transform extracts the subfunction specified by the user, wraps the
    subfunction in a Pmap op, and inserts the necessary collective communication
    ops to facilitate distributed execution.

    The user specifies which input values to partition between each device as well
    as the dimension to partition for each input. The specified input values are
    scattered between each device, while the remaining input values are broadcasted.
    The outputs of the Pmap operator will be aggregated using the user-specified
    reduction functions.

    Data parallelism is achieved by partitioning the input data along the batch
    dimension and replicating the weights. Horizontal parallelism is achieved by
    partitioning the weights and replicating the input data.

    This transform does not guarantee semantic equivalance; the user can
    optionally submit a verification function which ensures the specified
    subfunction is valid before applying the transform. Furthermore, we assume
    that the specified subfunction spans a logically contiguous set of ops.

    Args:
      function: The DistIR function to transform.
      ops: The list of ops that span the subfunction to map across devices.
      input_dims: A map from input value to partition dimension.
      reduction_params: A map from output value to a map of reduction op params.
      devices: The devices over which to map the model.
      verify_fn: A Python function that accepts as input the DistIR function to transform,
      the input_dims, and the reduction_params. Returns True if the specified function
      can be transformed, and False otherwise.

    Returns:
      The transformed function.
    """

    transformed_function = FunctionMaker()
    subfunction = function.get_subfunction(ops)
    if verify_fn is not None:
        if not verify_fn(subfunction, input_dims, reduction_params):
            return None

    # Either scatter or broadcast each input value depending on what the user
    # has requested.
    # TODO: Add explicit Send ops if the source device is not one of the
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
                if input_value in input_dims:
                    vs = transformed_function.add_op(
                        "MPIScatterToTupleType",
                        name=f"MPIScatter/{v.name}",
                        inputs=[v],
                        attributes={
                            "devices": devices,
                            "dim": input_dims[input_value],
                        },
                        output_names=[f"{v.name}s"],
                    )
                else:
                    vs = transformed_function.add_op(
                        "MPIBroadcastToTupleType",
                        name=f"MPIBroadcast/{v.name}",
                        inputs=[v],
                        attributes={"devices": devices},
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
                attributes={
                    "devices": devices,
                    "device_var": Device.get_new_device_variable("gpu"),
                },
                subfunctions=[subfunction],
                output_names=pmap_output_names,
            )

            if not isinstance(pmap_output_values, tuple):
                pmap_output_values = (pmap_output_values,)

            # Add reduction operators to collect output values from each device.
            # TODO: Add explicit Send ops if the destination device is not one of the
            #       source devices.
            for i, output_value in enumerate(subfunction.outputs):
                reduction_op_type = reduction_params[output_value]["op_type"]
                if reduction_op_type == "MPIAllreduce":
                    pmap_output = transformed_function.add_op(
                        "MPIAllreduceFromTupleType",
                        name=f"Allreduce/{output_value.name}",
                        inputs=[pmap_output_values[i]],
                        output_names=[f"{output_value.name}s"],
                    )
                elif reduction_op_type == "MPIReduce":
                    device = reduction_params[output_value]["device"]
                    pmap_output = transformed_function.add_op(
                        "MPIReduceFromTupleType",
                        name=f"MPIReduce/{output_value.name}",
                        attributes={"device": device},
                        inputs=[pmap_output_values[i]],
                        output_names=[f"{output_value.name}s"],
                    )
                elif reduction_op_type == "MPIGather":
                    dim = reduction_params[output_value]["dim"]
                    device = reduction_params[output_value]["device"]
                    pmap_output = transformed_function.add_op(
                        "MPIGatherFromTupleType",
                        name=f"MPIGather/{output_value.name}",
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
