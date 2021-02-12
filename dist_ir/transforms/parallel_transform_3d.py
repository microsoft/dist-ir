import re

from collections import defaultdict

from ..ir.function import FunctionMaker
from ..ir.op import Op
from ..ir import Device, cpprint
from .pipedream_scheduler import PipeDreamScheduler


def _scatter_value(v, function, dim, devices, parallelism_level):
    output_names = [f"{v.name}_{parallelism_level}_{i}" for i in range(len(devices))]
    return function.add_op(
        "MPIScatter",
        inputs=[v],
        attributes={"dim": dim, "devices": devices},
        output_names=output_names,
    )


def _broadcast_value(v, function, devices, parallelism_level):
    output_names = [f"{v.name}_{parallelism_level}_{i}" for i in range(len(devices))]
    return function.add_op(
        "MPIBroadcast",
        inputs=[v],
        attributes={"devices": devices},
        output_names=output_names,
    )


def _split_value(v, function, num_splits, parallelism_level):
    assert parallelism_level == "pp"
    output_names = [f"{v.name}_{parallelism_level}_{i}" for i in range(num_splits)]
    return function.add_op(
        "Split",
        inputs=[v],
        attributes={"dim": 0, "num_splits": num_splits},
        output_names=output_names,
    )


def _gather_values(vs, function, dim, device, output_name):
    return function.add_op(
        "MPIGather",
        inputs=[_join_tuple(vs, function, output_name=output_name)],
        attributes={"dim": dim, "device": device},
        output_names=[output_name],
    )


def _allgather_values(
    vs, function, dim, output_name, expand_tuple=False, expanded_output_names=None
):
    gathered_vs = function.add_op(
        "Allgather",
        attributes={"dim": dim},
        inputs=[_join_tuple(vs, function, output_name=output_name)],
        output_names=[output_name],
    )
    if expand_tuple:
        return _expand_tuple(
            gathered_vs, function, len(tuple(vs)), output_names=expanded_output_names
        )
    else:
        return gathered_vs


def _allreduce_values(vs, function, output_names):
    return function.add_op(
        "MPIAllreduce",
        inputs=vs,
        output_names=output_names,
    )


def _send_value(v, device, function):
    return function.add_op(
        "Send",
        inputs=[v],
        attributes={"device": device},
        output_names=[f"{v.name}@{device.device_id}"],
    )


def _add_values(v1, v2, function, output_name):
    return function.add_op("Add", inputs=[v1, v2], output_names=[output_name])


def _concat_values(v1, v2, function, dim, output_name):
    return function.add_op(
        "Concat", inputs=[v1, v2], attributes={"dim": dim}, output_names=[output_name]
    )


def _mpi_reduce_values(vs, function, device, output_name):
    return function.add_op(
        "MPIReduce",
        inputs=vs,
        attributes={"device": device},
        output_names=[output_name],
    )


def _mpi_gather_values(vs, function, dim, device, output_name):
    return function.add_op(
        "MPIGather",
        inputs=vs,
        attributes={"dim": dim, "device": device},
        output_names=[output_name],
    )


def _identity(v, function, output_name):
    return function.add_op("Identity", inputs=[v], output_names=[output_name])


def _partition_inputs_dp(function, device_tree):
    """Partitions inputs using data parallelism."""

    x, z, weights = function.inputs[0], function.inputs[1], function.inputs[2:]
    device_tree_root = tuple(device_tree.keys())[0]
    dp_devices = tuple(device_tree[device_tree_root].keys())
    dp_inputs = {}
    if len(dp_devices) > 1:
        dp_inputs[x] = _scatter_value(
            x, function, dim=0, devices=dp_devices, parallelism_level="dp"
        )
        dp_inputs[z] = _scatter_value(
            z, function, dim=0, devices=dp_devices, parallelism_level="dp"
        )
        for weight in weights:
            dp_inputs[weight] = _broadcast_value(
                weight, function, devices=dp_devices, parallelism_level="dp"
            )
    else:
        dp_inputs[x] = [x]
        dp_inputs[z] = [z]
        for weight in weights:
            dp_inputs[weight] = [weight]
    return dp_inputs


def _partition_inputs_hp(function, device_tree, dp_inputs):
    """Partitions inputs using horizontal parallelism."""
    x, z, weights = function.inputs[0], function.inputs[1], function.inputs[2:]
    device_tree_root = tuple(device_tree.keys())[0]
    dp_devices = tuple(device_tree[device_tree_root].keys())
    hp_inputs = {}
    for i, dp_device in enumerate(dp_devices):
        hp_devices = tuple(device_tree[device_tree_root][dp_device])
        if len(hp_devices) > 1:
            hp_inputs[dp_inputs[x][i]] = _broadcast_value(
                dp_inputs[x][i],
                function,
                devices=hp_devices,
                parallelism_level="hp",
            )
            hp_inputs[dp_inputs[z][i]] = _broadcast_value(
                dp_inputs[z][i],
                function,
                devices=hp_devices,
                parallelism_level="hp",
            )
            for j, weight in enumerate(weights):
                dim = (j + 1) % 2
                # dim = 1
                hp_inputs[dp_inputs[weight][i]] = _scatter_value(
                    dp_inputs[weight][i],
                    function,
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
    function,
    device_tree,
    dp_inputs,
    hp_inputs,
    num_microbatches,
):
    """Partitions inputs using pipeline parallelism."""
    x, z, weights = function.inputs[0], function.inputs[1], function.inputs[2:]
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
                    function,
                    num_splits=num_microbatches,
                    parallelism_level="pp",
                )
                pp_inputs[hp_z] = _split_value(
                    hp_z,
                    function,
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
    dp_inputs = _partition_inputs_dp(transformed_function, device_tree)
    hp_inputs = _partition_inputs_hp(transformed_function, device_tree, dp_inputs)
    pp_inputs = _partition_inputs_pp(
        transformed_function,
        device_tree,
        dp_inputs,
        hp_inputs,
        num_microbatches,
    )

    device_tree_root = tuple(device_tree.keys())[0]
    dp_outputs = defaultdict(list)
    for i, dp_device in enumerate(device_tree[device_tree_root]):
        pp_schedules = []
        hp_devices = device_tree[device_tree_root][dp_device]
        for j, hp_device in enumerate(hp_devices):
            pp_devices = device_tree[device_tree_root][dp_device][hp_device]
            partition_map = _pipeline_parallel_partition(args, function, pp_devices)
            scheduler = PipeDreamScheduler(num_microbatches)
            schedule = scheduler.schedule(function, partition_map)
            pp_schedules.append(schedule)

        output_map = defaultdict(lambda: defaultdict(dict))
        matmul_counter = defaultdict(lambda: 0)
        # Jointly iterate through all the schedules, timestep by timestep.
        for timesteps in zip(*pp_schedules):
            # For a given set of timesteps, iterate through in order of matching
            # horizontal parallel devices.
            for devices in zip(*tuple(sorted(ts.keys()) for ts in timesteps)):
                # Verify that for this group of horizontal parallel devices the
                # corresponding pipeline parallel stage is exactly the same.
                assert (
                    len(set(ts[device] for ts, device in zip(timesteps, devices))) == 1
                )
                stage, microbatch_id = timesteps[0][devices[0]]
                for op in stage.ops:
                    # Collect inputs for this op.
                    for j, device in enumerate(devices):
                        input_values = []
                        input_devices = []
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
                                output_value, output_device = output_map[j][
                                    microbatch_id
                                ][inp]
                                input_values.append(output_value)
                                input_devices.append(output_device)
                        # Add the op once for each device to the transformed function.
                        transformed_outputs = transformed_function.add_op(
                            op.op_type,
                            inputs=input_values,
                            attributes=op.attributes,
                            output_names=[
                                f"{v.name}_dp_{i}_hp_{j}_pp_{microbatch_id}_device_{device}"
                                for v in op.outputs
                            ],
                        )
                        if not isinstance(transformed_outputs, tuple):
                            transformed_outputs = (transformed_outputs,)
                        for output, transformed_output in zip(
                            op.outputs, transformed_outputs
                        ):
                            assert output not in output_map[j][microbatch_id]
                            output_map[j][microbatch_id][output] = (
                                transformed_output,
                                device,
                            )
                    # TODO: Remove debug code
                    j = None
                    device = None

                    # Aggregate horizontal parallel outputs.
                    if op.op_type == "MatMul" or op.op_type == "MatMulGrad":
                        matmul_counter[microbatch_id] += 1
                        if matmul_counter[microbatch_id] % 2 == 0:
                            for output in op.outputs:
                                if "dw" in output.name:
                                    # Weight gradients do not need to be aggregated
                                    # across model parallel partitions.
                                    continue
                                print(
                                    f"Doing horizontal parallel reduction for microbatch {microbatch_id} for "
                                    f"{tuple(output_map[j][microbatch_id][output][0] for j in range(len(devices)))}"
                                )
                                reduced_outputs = _allreduce_values(
                                    tuple(
                                        output_map[j][microbatch_id][output][0]
                                        for j in range(len(devices))
                                    ),
                                    transformed_function,
                                    output_names=[
                                        f"{output.name}_dp_{i}_hp_all_pp_{microbatch_id}_device_{device.device_id}"
                                        for j, device in enumerate(devices)
                                    ],
                                )
                                assert len(reduced_outputs) == len(devices)
                                for k, (d, reduced_output) in enumerate(
                                    zip(devices, reduced_outputs)
                                ):
                                    output_map[k][microbatch_id][output] = (
                                        reduced_output,
                                        d,
                                    )

                    # Aggregate pipeline parallel outputs.
                    for output in op.outputs:
                        if output in function.outputs:
                            for j, device in enumerate(devices):
                                mb_k_output, mb_k_device = output_map[j][microbatch_id][
                                    output
                                ]
                                assert mb_k_device == device
                                match = re.search("hp\_(.*)\_pp", mb_k_output.name)
                                hp_level = match.group(1)
                                if microbatch_id == 0:
                                    output_map[j]["all"][output] = (
                                        _identity(
                                            mb_k_output,
                                            transformed_function,
                                            f"{output.name}_dp_{i}_hp_{hp_level}_pp_all_device_{mb_k_device.device_id}",
                                        ),
                                        mb_k_device,
                                    )
                                else:
                                    assert output in output_map[j]["all"]
                                    mb_all_output, mb_all_device = output_map[j]["all"][
                                        output
                                    ]
                                    assert mb_all_device == device
                                    assert (
                                        re.search(
                                            "hp\_(.*)\_pp", mb_all_output.name
                                        ).group(1)
                                        == hp_level
                                    )
                                    print(
                                        f"Doing pipeline parallel aggregation for {mb_all_output} "
                                        f"and {mb_k_output} on device {device.device_id}"
                                    )
                                    if "dw" in output.name:
                                        output_map[j]["all"][output] = (
                                            _add_values(
                                                mb_all_output,
                                                mb_k_output,
                                                transformed_function,
                                                output_name=f"{output.name}_dp_{i}_hp_{hp_level}_pp_all_device_{mb_all_device.device_id}",
                                            ),
                                            mb_all_device,
                                        )
                                    else:
                                        output_map[j]["all"][output] = (
                                            _concat_values(
                                                mb_all_output,
                                                mb_k_output,
                                                transformed_function,
                                                dim=0,
                                                output_name=f"{output.name}_dp_{i}_hp_{hp_level}_pp_all_device_{mb_all_device.device_id}",
                                            ),
                                            mb_all_device,
                                        )
        # Collect the pipeline-parallel aggregated function outputs to do data parallel aggregation.
        for output in function.outputs:
            if "dw" in output.name:
                print(
                    f"Concatenating horizontal parallel values "
                    f"{tuple(output_map[j]['all'][output][0] for j in output_map)}"
                )
                match = re.search("dw(.)", output.name)
                weight_id = ord(match.group(1)) - ord("A")
                dim = (weight_id + 1) % 2
                gathered_gradient = _mpi_gather_values(
                    tuple(output_map[j]["all"][output][0] for j in output_map),
                    transformed_function,
                    dim=dim,
                    device=device_tree_root,
                    output_name=f"{output.name}_dp_{i}_hp_all_pp_all_mb_device_{device_tree_root.device_id}",
                )
                dp_outputs[output].append(gathered_gradient)
            else:
                for j in output_map:
                    value, _ = output_map[j]["all"][output]
                    dp_outputs[output].append(value)
    for output in dp_outputs:
        print(f"Doing data parallel reduction for {dp_outputs[output]}")
        if "dw" in output.name:
            _mpi_reduce_values(
                dp_outputs[output],
                transformed_function,
                device=device_tree_root,
                output_name=output.name,
            )
        else:
            _mpi_gather_values(
                dp_outputs[output],
                transformed_function,
                dim=0,
                device=device_tree_root,
                output_name=output.name,
            )
    return transformed_function.finalize()
