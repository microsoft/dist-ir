from collections import defaultdict

from ..ir.function import FunctionMaker
from ..ir.op import Op
from ..ir import Device, cpprint
from .pipedream_scheduler import PipeDreamScheduler


def _expand_tuple(vs, function, size, name, parallelism_level):
    return [
        function.add_op(
            "Select",
            inputs=[vs],
            attributes={"dim": i},
            output_names=[f"{name}_{parallelism_level}_{i}"],
        )
        for i in range(size)
    ]


def _scatter_value(v, function, dim, devices, parallelism_level):
    vs = function.add_op(
        "Scatter",
        inputs=[v],
        attributes={"dim": dim, "devices": devices},
        output_names=[f"{v.name}s_{parallelism_level}"],
    )
    return _expand_tuple(vs, function, len(devices), v.name, parallelism_level)


def _broadcast_value(v, function, devices, parallelism_level):
    vs = function.add_op(
        "Broadcast",
        inputs=[v],
        attributes={"devices": devices},
        output_names=[f"{v.name}s_{parallelism_level}"],
    )
    return _expand_tuple(vs, function, len(devices), v.name, parallelism_level)


def _split_value(v, function, num_splits, parallelism_level):
    assert parallelism_level == "pp"
    vs = function.add_op(
        "Split",
        inputs=[v],
        attributes={"dim": 0, "num_splits": num_splits},
        output_names=[f"{v.name}s_{parallelism_level}"],
    )
    return _expand_tuple(vs, function, num_splits, v.name, parallelism_level)


def _send_value(v, device, function):
    return function.add_op(
        "Send",
        inputs=[v],
        attributes={"device": device},
        output_names=[f"{v.name}@{device.device_id}"],
    )


def _partition_inputs_dp(transformed_function, device_tree, x, z, weights):
    """Partitions inputs using data parallelism."""

    device_tree_root = tuple(device_tree.keys())[0]
    dp_devices = tuple(device_tree[device_tree_root].keys())
    dp_inputs = {}
    if len(dp_devices) > 1:
        dp_inputs[x] = _scatter_value(
            x, transformed_function, dim=0, devices=dp_devices, parallelism_level="dp"
        )
        dp_inputs[z] = _scatter_value(
            z, transformed_function, dim=0, devices=dp_devices, parallelism_level="dp"
        )
        for weight in weights:
            dp_inputs[weight] = _broadcast_value(
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
            hp_inputs[dp_inputs[x][i]] = _broadcast_value(
                dp_inputs[x][i],
                transformed_function,
                devices=hp_devices,
                parallelism_level="hp",
            )
            hp_inputs[dp_inputs[z][i]] = _broadcast_value(
                dp_inputs[z][i],
                transformed_function,
                devices=hp_devices,
                parallelism_level="hp",
            )
            for j, weight in enumerate(weights):
                dim = (j + 1) % 2
                hp_inputs[dp_inputs[weight][i]] = _scatter_value(
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
                pp_inputs[hp_x] = _split_value(
                    hp_x,
                    transformed_function,
                    num_splits=num_microbatches,
                    parallelism_level="pp",
                )
                pp_inputs[hp_z] = _split_value(
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
    return pp_inputs


def _pipeline_parallel_partition(args, function, devices):
    num_blocks = args.num_hidden_layers + 1
    assert num_blocks % args.pp_degree == 0
    num_blocks_per_device = num_blocks // args.pp_degree
    partition_map = {}
    for i, device in enumerate(devices):
        fwd_start = i * num_blocks_per_device * 2
        fwd_end = (i + 1) * num_blocks_per_device * 2 + (
            2 if i == args.pp_degree - 1 else 0
        )
        bwd_start = len(function.ops) - ((i + 1) * num_blocks_per_device * 2)
        bwd_end = bwd_start + num_blocks_per_device * 2
        fwd_stage = function.get_subfunction(
            function.ops[fwd_start:fwd_end],
            name=f"fwd_stage{i}",
        )
        bwd_stage = function.get_subfunction(
            function.ops[bwd_start:bwd_end],
            name=f"bwd_stage{i}",
        )
        partition_map[fwd_stage] = device
        partition_map[bwd_stage] = device
    return partition_map


def parallel_transform_3d(args, function, device_tree, num_microbatches):
    transformed_function = FunctionMaker(name=function.name)

    # Add inputs to the transformed function.
    transformed_inputs = {}
    for inp in function.inputs:
        v = transformed_function.add_input_value(inp.name, inp.type)
        transformed_inputs[inp] = v

    # Partition inputs across each parallelism dimension.
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

    # Instantiate the function for each subgroup of devices.
    device_tree_root = tuple(device_tree.keys())[0]
    for i, dp_device in enumerate(device_tree[device_tree_root]):
        for j, hp_device in enumerate(device_tree[device_tree_root][dp_device]):
            output_map = {}
            pp_devices = device_tree[device_tree_root][dp_device][hp_device]
            partition_map = _pipeline_parallel_partition(args, function, pp_devices)
            scheduler = PipeDreamScheduler(num_microbatches)
            schedule = scheduler.schedule(function, partition_map)
            for timestep in schedule:
                for device in timestep:
                    stage, microbatch_id = timestep[device]
                    for op in stage.ops:
                        input_values = []
                        input_devices = []
                        # Collect the transformed function's version of the op's inputs.
                        # If an input value is a function input value, find the corresponding
                        # partitioned input value. Otherwise the input value must be an intermediate
                        # output from an upstream op.
                        for inp in op.inputs:
                            if inp in function.inputs:
                                v = transformed_inputs[inp]
                                dp_v = dp_inputs[v][i]
                                hp_v = hp_inputs[dp_v][j]
                                if (
                                    inp == function.inputs[0]
                                    or inp == function.inputs[1]
                                ):
                                    pp_v = pp_inputs[hp_v][microbatch_id]
                                else:
                                    pp_v = pp_inputs[hp_v][0]
                                input_values.append(pp_v)
                                input_devices.append(hp_device)
                            else:
                                output_value, output_device = output_map[inp]
                                input_values.append(output_value)
                                input_devices.append(output_device)
                        # Forward any inputs that are not on the correct device.
                        # TODO: Don't send weights multiple times across microbatches.
                        for k, (input_value, input_device) in enumerate(
                            zip(input_values, input_devices)
                        ):
                            if input_device != device:
                                input_values[k] = _send_value(
                                    input_values[k], device, transformed_function
                                )
                        transformed_outputs = transformed_function.add_op(
                            op.op_type,
                            inputs=input_values,
                            attributes=op.attributes,
                            output_names=[v.name for v in op.outputs],
                        )
                        if not isinstance(transformed_outputs, tuple):
                            transformed_outputs = (transformed_outputs,)
                        for output, transformed_output in zip(
                            op.outputs, transformed_outputs
                        ):
                            output_map[output] = (transformed_output, device)
                        # TODO: Aggregate outputs for pipeline parallelism
                        # TODO: Aggregate outputs for horizontal parallelism
                        # TODO: Aggregate outputs for data parallelism
    return transformed_function.finalize()
