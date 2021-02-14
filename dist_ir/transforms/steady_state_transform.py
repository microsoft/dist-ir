from ..ir.function import FunctionMaker


def steady_state_transform(function):
    """Removes any send ops from input values to isolate steady state behavior."""

    done = False
    iteration = 0
    inv_value_maps = []
    global_inv_value_map = {}
    while not done:
        print(f"Iteration {iteration}")
        done = True
        transformed_function = FunctionMaker()
        value_map = {}
        for op in function.ops:
            if all(inp in function.inputs for inp in op.inputs) and (
                op.op_type == "MPIBroadcast"
                or op.op_type == "MPIScatter"
                or op.op_type == "Split"
            ):
                for output in op.outputs:
                    v = transformed_function.add_input_value(output.name, output.type)
                    value_map[output] = v
                done = False
                continue
            inputs = []
            for inp in op.inputs:
                if inp not in value_map:
                    v = transformed_function.add_input_value(inp.name, inp.type)
                    value_map[inp] = v
                inputs.append(value_map[inp])
            outputs = transformed_function.add_op(
                op.op_type,
                inputs=inputs,
                attributes=op.attributes,
                output_names=[output.name for output in op.outputs],
            )
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for output, transformed_output in zip(op.outputs, outputs):
                value_map[output] = transformed_output
        inv_value_maps.append({v: k for k, v in value_map.items()})
        for inp in transformed_function.inputs:
            v = inp
            for inv_value_map in inv_value_maps[::-1]:
                v = inv_value_map[v]
            global_inv_value_map[inp] = v
        iteration += 1
        function = transformed_function.finalize()
    typed_input_values = [
        global_inv_value_map[inp] for inp in transformed_function.inputs
    ]
    for v in typed_input_values:
        assert v.type is not None
    return function, typed_input_values
