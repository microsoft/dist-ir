from collections import defaultdict

from dist_ir.ir import FunctionMaker


def _mpi_allgather_values(vs, function, dim):
    return function.add_op(
        "MPIAllgather",
        inputs=vs,
        attributes={"dim": dim},
        output_names=[f"{v.name}_gathered" for v in vs],
    )


def _mpi_allreduce_values(vs, function):
    return function.add_op(
        "MPIAllreduce",
        inputs=vs,
        output_names=[f"{v.name}_reduced" for v in vs],
    )


def _mpi_broadcast_value(v, function, devices):
    return function.add_op(
        "MPIBroadcast",
        inputs=[v],
        attributes={"devices": devices},
        output_names=[f"{v.name}_{d.device_id}" for d in devices],
    )


def _mpi_scatter_value(v, function, dim, devices):
    return function.add_op(
        "MPIScatter",
        inputs=[v],
        attributes={"dim": dim, "devices": devices},
        output_names=[f"{v.name}_{d.device_id}" for d in devices],
    )


def spmd_transform(function, input_dims, reduction_params, devices):
    transformed_function = FunctionMaker()

    value_map = defaultdict(list)
    for inp in function.inputs:
        v = transformed_function.add_input_value(inp.name, inp.type)
        if inp in input_dims:
            dim = input_dims[inp]
            vs = _mpi_scatter_value(v, transformed_function, dim, devices)
        else:
            vs = _mpi_broadcast_value(v, transformed_function, devices)
        value_map[inp] = list(vs)

    for i, d in enumerate(devices):
        for op in function.ops:
            inputs = [value_map[inp][i] for inp in op.inputs]
            outputs = transformed_function.add_op(
                op.op_type,
                inputs=inputs,
                attributes=op.attributes,
                output_names=[f"{output.name}_{d.device_id}" for output in op.outputs],
            )
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for output, v in zip(op.outputs, outputs):
                value_map[output].append(v)

    for output in reduction_params:
        op_type = reduction_params[output]["op_type"]
        if op_type == "MPIAllreduce":
            _mpi_allreduce_values(value_map[output], transformed_function)
        elif op_type == "MPIAllgather":
            _mpi_allgather_values(
                value_map[output],
                transformed_function,
                dim=dim,
            )
        else:
            raise ValueError(
                f"Unknown reduction op type {op_type} for output {output.name}"
            )

    return transformed_function.finalize()
