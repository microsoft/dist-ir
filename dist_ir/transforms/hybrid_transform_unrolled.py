from collections import defaultdict

from ..ir.function import FunctionMaker
from ..ir.op import Op
from ..ir import Device, cpprint
from .shard_transform import shard_transform
from .pipeline_parallel_transform import PipelineParallelTransform


def _reduce_output(
    transformed_function,
    orig_output,
    transformed_outputs,
    reduction_params,
    device,
    parallelism_level,
):
    transformed_output = transformed_function.add_op(
        "Join",
        inputs=transformed_outputs,
        output_names=[f"{orig_output.name}s_{parallelism_level}"],
    )
    if reduction_params["op_type"] == "MPIReduce":
        reduced_output = transformed_function.add_op(
            "MPIReduce",
            attributes={"device": device},
            inputs=[transformed_output],
            output_names=[f"{orig_output.name}_{parallelism_level}"],
        )
    elif reduction_params["op_type"] == "MPIGather":
        dim = reduction_params["dim"]
        reduced_output = transformed_function.add_op(
            "MPIGather",
            attributes={"dim": dim, "device": device},
            inputs=[transformed_output],
            output_names=[f"{orig_output.name}_{parallelism_level}"],
        )
    return reduced_output


def _partition_input_dp(inp, transformed_function, dp_config):
    v = transformed_function.add_input_value(inp.name, inp.type)
    dp_vs = []
    if inp in dp_config["input_dims"]:
        distributed_v = transformed_function.add_op(
            "Scatter",
            inputs=[v],
            attributes={
                "dim": dp_config["input_dims"][inp],
                "devices": dp_config["devices"],
            },
            output_names=[f"{v.name}is_dp"],
        )
    else:
        distributed_v = transformed_function.add_op(
            "Broadcast",
            name=f"Broadcast/{inp.name}",
            inputs=[v],
            attributes={"devices": dp_config["devices"]},
            output_names=[f"{v.name}is_dp"],
        )
    dp_vs = [
        transformed_function.add_op(
            "Select",
            inputs=[distributed_v],
            attributes={"dim": i},
            output_names=[f"{v.name}_dp_{device.device_id}"],
        )
        for i, device in enumerate(dp_config["devices"])
    ]
    return dp_vs


def hybrid_transform_unrolled(function, dp_config, hp_config, pp_config=None):
    transformed_function = FunctionMaker(name=function.name)
    input_map = {}
    for inp in function.inputs:
        v = transformed_function.add_input_value(inp.name, inp.type)
        dp_vs = _partition_input_dp(inp, transformed_function, dp_config)
        hp_vs = []
        for i, dp_v in enumerate(dp_vs):
            if inp in hp_config["input_dims"]:
                distributed_v = transformed_function.add_op(
                    "Scatter",
                    inputs=[dp_v],
                    attributes={
                        "dim": hp_config["input_dims"][inp],
                        "devices": hp_config["devices"][dp_config["devices"][i]],
                    },
                    output_names=[f"{v.name}is_hp"],
                )
            else:
                distributed_v = transformed_function.add_op(
                    "Broadcast",
                    inputs=[dp_v],
                    attributes={
                        "devices": hp_config["devices"][dp_config["devices"][i]],
                    },
                    output_names=[f"{v.name}is_hp"],
                )

            hp_vs += [
                transformed_function.add_op(
                    "Select",
                    inputs=[distributed_v],
                    attributes={"dim": j},
                    output_names=[f"{v.name}_hp_{device.device_id}"],
                )
                for j, device in enumerate(
                    hp_config["devices"][dp_config["devices"][i]]
                )
            ]

        # TODO: Handle pipeline parallelism
        input_map[inp] = hp_vs

    dp_outputs_to_reduce = defaultdict(list)
    for i in range(len(dp_config["devices"])):
        hp_outputs_to_reduce = defaultdict(list)
        for j, device in enumerate(hp_config["devices"][dp_config["devices"][i]]):
            output_map = {}
            for op in function.ops:
                inputs = []
                for inp in op.inputs:
                    if inp in input_map:
                        inputs.append(input_map[inp][i * len(dp_config["devices"]) + j])
                    elif inp in output_map:
                        inputs.append(output_map[inp])
                    else:
                        raise ValueError(f"Could not find input {inp}")
                outputs = transformed_function.add_op(
                    op.op_type,
                    op.name,
                    inputs=inputs,
                    attributes=op.attributes,
                    output_names=[
                        f"{output.name}_{device.device_id}" for output in op.outputs
                    ],
                )
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                for k, output in enumerate(outputs):
                    output_map[op.outputs[k]] = output
                    if op.outputs[k] in hp_config["reduction_params"]:
                        hp_outputs_to_reduce[op.outputs[k]].append(output)
        for hp_output in hp_outputs_to_reduce:
            reduced_output = _reduce_output(
                transformed_function,
                hp_output,
                hp_outputs_to_reduce[hp_output],
                hp_config["reduction_params"][hp_output],
                dp_config["devices"][i],
                "hp",
            )
            dp_outputs_to_reduce[hp_output].append(reduced_output)
    for dp_output in dp_outputs_to_reduce:
        _reduce_output(
            transformed_function,
            dp_output,
            dp_outputs_to_reduce[dp_output],
            dp_config["reduction_params"][dp_output],
            dp_config["reduction_params"][dp_output]["device"],
            "dp",
        )
    return transformed_function.finalize()
