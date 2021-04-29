from ..ir import Op
from ..ir.function import FunctionMaker
from .sanitize_attributes_transform import (
    sanitize_unhashable_attributes,
    restore_unhashable_attributes,
)


def filter_transform(
    function,
    filter_set=set(["MPIBroadcast", "MPIScatter", "Send", "Split"]),
    exception_set=set(),
):
    """Filters ops of specified op type(s) from the given function.

    Removes ops from the given function and constructs a new function with the
    removed ops' output values as input values. Runs this procedure iteratively
    until reaching fixed point.

    This transform can be used to isolate steady state behavior after applying
    parallel transforms.

    Note that the filter set is a set of op types while the exception set is a set
    of specific Values. This is useful, for example, in the case where we want to
    only filter sends of weights and not input values when using pipeline parallelism.

    Args:
      function: The function to transform.
      filter_set: The set of op types to remove.
      exception_set: A set of input values to keep regardless
                     of the specified filter set.

    Returns:
      The transformed function.
    """

    function, attribute_map = sanitize_unhashable_attributes(function)

    done = False
    inv_value_maps = []
    global_inv_value_map = {}
    for inp in function.inputs:
        global_inv_value_map[inp] = inp
    while not done:
        done = True
        transformed_function = FunctionMaker()
        value_map = {}
        input_consumers = set()
        function_inputs_set = set(function.inputs)
        for inp in function.inputs:
            consumers = function.consumers[inp]
            for consumer in consumers:
                if set(consumer.inputs).issubset(function_inputs_set):
                    input_consumers.add(consumer)
        for op in function.ops:
            if (
                op in input_consumers
                and op.op_type in filter_set
                and not any(
                    global_inv_value_map[inp] in exception_set for inp in op.inputs
                )
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
            new_op = Op(
                name=op.name,
                op_type=op.op_type,
                inputs=tuple(inputs),
                attributes=op.attributes,
                subfunctions=op.subfunctions,
                output_names=tuple(output.name for output in op.outputs),
                output_types=tuple(output.type for output in op.outputs),
            )
            transformed_function.ops.append(new_op)
            for output, transformed_output in zip(op.outputs, new_op.outputs):
                value_map[output] = transformed_output
        inv_value_maps.append({v: k for k, v in value_map.items()})
        for inp in transformed_function.inputs:
            v = inp
            for inv_value_map in inv_value_maps[::-1]:
                v = inv_value_map[v]
            global_inv_value_map[inp] = v
        assert len(transformed_function.ops) <= len(function.ops)
        function = restore_unhashable_attributes(transformed_function, attribute_map)
        function = transformed_function.finalize()
    typed_input_values = [
        global_inv_value_map[inp] for inp in transformed_function.inputs
    ]
    for v in typed_input_values:
        if v.type is None:
            raise ValueError(f"Input value {v} has no type!")
    return function, typed_input_values
