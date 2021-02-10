from collections import defaultdict

from ..ir.function import FunctionMaker
from ..ir.op import Op
from ..ir import Device, cpprint
from .shard_transform import shard_transform
from .pipeline_parallel_transform import PipelineParallelTransform


def _expand_tuple(vs, transformed_function, size, name, parallelism_level):
    return [
        transformed_function.add_op(
            "Select",
            inputs=[vs],
            attributes={"dim": i},
            output_names=[f"{name}_{parallelism_level}_{i}"],
        )
        for i in range(size)
    ]


def _scatter_input(v, transformed_function, dim, devices, parallelism_level):
    vs = transformed_function.add_op(
        "Scatter",
        inputs=[v],
        attributes={"dim": dim, "devices": devices},
        output_names=[f"{v.name}s_{parallelism_level}"],
    )
    return _expand_tuple(
        vs, transformed_function, len(devices), v.name, parallelism_level
    )


def _broadcast_input(v, transformed_function, devices, parallelism_level):
    vs = transformed_function.add_op(
        "Broadcast",
        inputs=[v],
        attributes={"devices": devices},
        output_names=[f"{v.name}s_{parallelism_level}"],
    )
    return _expand_tuple(
        vs, transformed_function, len(devices), v.name, parallelism_level
    )


def _split_input(v, transformed_function, num_splits, parallelism_level):
    assert parallelism_level == "pp"
    vs = transformed_function.add_op(
        "Split",
        inputs=[v],
        attributes={"dim": 0, "num_splits": num_splits},
        output_names=[f"{v.name}s_{parallelism_level}"],
    )
    return _expand_tuple(
        vs, transformed_function, num_splits, v.name, parallelism_level
    )


def _partition_inputs_dp(transformed_function, device_tree, x, z, weights):
    """Partitions inputs using data parallelism."""

    device_tree_root = tuple(device_tree.keys())[0]
    dp_devices = tuple(device_tree[device_tree_root].keys())
    dp_inputs = {}
    if len(dp_devices) > 1:
        dp_inputs[x] = _scatter_input(
            x, transformed_function, dim=0, devices=dp_devices, parallelism_level="dp"
        )
        dp_inputs[z] = _scatter_input(
            z, transformed_function, dim=0, devices=dp_devices, parallelism_level="dp"
        )
        for weight in weights:
            dp_inputs[weight] = _broadcast_input(
                weight, transformed_function, devices=dp_devices, parallelism_level="dp"
            )
    else:
        dp_inputs[x] = [x]
        dp_inputs[z] = [z]
        for weight in weights:
            dp_inputs[weight] = [weight]
    return dp_inputs


def _partition_inputs_hp(transformed_function, device_tree, dp_inputs, x, z, weights):
    """Partitions inputs using horizontal parallelism."""
    device_tree_root = tuple(device_tree.keys())[0]
    dp_devices = tuple(device_tree[device_tree_root].keys())
    hp_inputs = {}
    for i, dp_device in enumerate(dp_devices):
        hp_devices = tuple(device_tree[device_tree_root][dp_device])
        if len(hp_devices) > 1:
            hp_inputs[dp_inputs[x][i]] = _broadcast_input(
                dp_inputs[x][i],
                transformed_function,
                devices=hp_devices,
                parallelism_level="hp",
            )
            hp_inputs[dp_inputs[z][i]] = _broadcast_input(
                dp_inputs[z][i],
                transformed_function,
                devices=hp_devices,
                parallelism_level="hp",
            )
            for j, weight in enumerate(weights):
                dim = (j + 1) % 2
                hp_inputs[dp_inputs[weight][i]] = _scatter_input(
                    dp_inputs[weight][i],
                    transformed_function,
                    dim=dim,
                    devices=hp_devices,
                    parallelism_level="hp",
                )
        else:
            hp_inputs[dp_inputs[x][i]] = [dp_inputs[x][i]]
            hp_inputs[dp_inputs[z][i]] = [dp_inputs[z][i]]
            for weight in weights:
                hp_inputs[dp_inputs[weight][i]] = [dp_inputs[weight][i]]
    return hp_inputs


def _partition_inputs_pp(
    transformed_function,
    device_tree,
    dp_inputs,
    hp_inputs,
    num_microbatches,
    x,
    z,
    weights,
):
    """Partitions inputs using pipeline parallelism."""
    device_tree_root = tuple(device_tree.keys())[0]
    dp_devices = tuple(device_tree[device_tree_root].keys())
    pp_inputs = {}
    for i, dp_device in enumerate(dp_devices):
        hp_devices = tuple(device_tree[device_tree_root][dp_device])
        for j, hp_device in enumerate(hp_devices):
            pp_devices = tuple(device_tree[device_tree_root][dp_device][hp_device])
            if len(pp_devices) > 1:
                hp_x = hp_inputs[dp_inputs[x][i]][j]
                hp_z = hp_inputs[dp_inputs[z][i]][j]
                pp_inputs[hp_x] = _split_input(
                    hp_x,
                    transformed_function,
                    num_splits=num_microbatches,
                    parallelism_level="pp",
                )
                pp_inputs[hp_z] = _split_input(
                    hp_z,
                    transformed_function,
                    num_splits=num_microbatches,
                    parallelism_level="pp",
                )
            else:
                pp_inputs[hp_inputs[x][j]] = [hp_inputs[x][j]]
                pp_inputs[hp_inputs[z][i]] = [hp_inputs[z][j]]
            for weight in weights:
                hp_weight = hp_inputs[dp_inputs[weight][i]][j]
                pp_inputs[hp_weight] = [hp_weight]


def parallel_transform_3d(function, device_tree, num_microbatches):
    transformed_function = FunctionMaker(name=function.name)

    # Add inputs to the transformed function.
    for inp in function.inputs:
        transformed_function.add_input_value(inp.name, inp.type)

    x, z, weights = (
        transformed_function.inputs[0],
        transformed_function.inputs[1],
        transformed_function.inputs[2:],
    )

    dp_inputs = _partition_inputs_dp(transformed_function, device_tree, x, z, weights)
    hp_inputs = _partition_inputs_hp(
        transformed_function, device_tree, dp_inputs, x, z, weights
    )
    pp_inputs = _partition_inputs_pp(
        transformed_function,
        device_tree,
        dp_inputs,
        hp_inputs,
        num_microbatches,
        x,
        z,
        weights,
    )
    return transformed_function
