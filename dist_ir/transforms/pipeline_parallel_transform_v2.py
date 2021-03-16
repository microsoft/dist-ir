from collections import defaultdict

from dist_ir.ir import FunctionMaker


def _add_values(v1, v2, transformed_function):
    return transformed_function.add_op("Add", inputs=[v1, v2], output_names=[v1.name])


def _concat_values(v1, v2, transformed_function, dim):
    return transformed_function.add_op(
        "Concat", inputs=[v1, v2], attributes={"dim": dim}, output_names=[v1.name]
    )


def _identity(v, transformed_function, output_name):
    return transformed_function.add_op(
        "Identity", inputs=[v], output_names=[output_name]
    )


def _send_value(v, transformed_function, device):
    return transformed_function.add_op(
        "Send",
        inputs=[v],
        attributes={"device": device},
        output_names=[f"{v.name}@{device.device_id}"],
    )


def _split_value(v, transformed_function, num_microbatches):
    return transformed_function.add_op(
        "Split",
        inputs=[v],
        attributes={"dim": 0, "num_splits": num_microbatches},
        output_names=[f"{v.name}_microbatch={i}" for i in range(num_microbatches)],
    )


def pipeline_parallel_transform_v2(
    function,
    op_to_stage_map,
    partitioned_device_map,
    batch_inputs,
    reduction_params,
    num_microbatches,
):
    transformed_function = FunctionMaker(name=f"{function.name}_pp")

    # Indexed by original function input and then microbatch id
    value_map = defaultdict(lambda: {})
    for inp in function.inputs:
        v = transformed_function.add_input_value(inp.name, inp.type)
        if inp in batch_inputs:
            vs = _split_value(v, transformed_function, num_microbatches)
        else:
            vs = tuple(v for _ in range(num_microbatches))
        assert len(vs) == num_microbatches
        for i, v in enumerate(vs):
            value_map[inp][i] = (v, inp.type.device)

    sent_values = set()
    aggregated_outputs = {}
    for i in range(num_microbatches):
        for op in function.ops:
            assert len(set([inp.type.device for inp in op.inputs])) == 1
            stage = op_to_stage_map[op]
            orig_device = op.inputs[0].type.device
            new_device = partitioned_device_map[orig_device][stage]
            for inp in op.inputs:
                orig_device = value_map[inp][i][1]
                if new_device != orig_device:
                    if i > 0 and inp in function.inputs and inp not in batch_inputs:
                        value_map[inp][i] = value_map[inp][0]
                    elif not (inp, i, orig_device, new_device) in sent_values:
                        value_map[inp][i] = (
                            _send_value(
                                value_map[inp][i][0], transformed_function, new_device
                            ),
                            new_device,
                        )
                        sent_values.add((inp, i, orig_device, new_device))
            inputs = [value_map[inp][i][0] for inp in op.inputs]
            outputs = transformed_function.add_op(
                op.op_type,
                inputs=inputs,
                attributes=op.attributes,
                output_names=[f"{output.name}_microbatch={i}" for output in op.outputs],
            )
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for output, v in zip(op.outputs, outputs):
                value_map[output][i] = (v, new_device)

                if output in function.outputs:
                    if i == 0:
                        aggregated_outputs[output] = (
                            _identity(
                                v,
                                transformed_function,
                                output_name=f"{output.name}_all",
                            ),
                            new_device,
                        )
                    else:
                        if reduction_params[output]["op_type"] == "Add":
                            aggregated_outputs[output] = (
                                _add_values(
                                    aggregated_outputs[output][0],
                                    v,
                                    transformed_function,
                                ),
                                new_device,
                            )
                        elif reduction_params[output]["op_type"] == "Concat":
                            dim = reduction_params[output]["dim"]
                            aggregated_outputs[output] = (
                                _concat_values(
                                    aggregated_outputs[output][0],
                                    v,
                                    transformed_function,
                                    dim,
                                ),
                                new_device,
                            )

    return transformed_function.finalize()
