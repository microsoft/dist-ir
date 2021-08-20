from collections import defaultdict, Hashable
from frozendict import frozendict
from itertools import chain
import math
import numpy as np
import logging
import re
import roundrobin


from ..ir import cpprint, Op
from ..ir.function import Function, FunctionMaker
from .pipedream_scheduler import PipeDreamScheduler
from .sanitize_attributes_transform import (
    sanitize_unhashable_attributes,
    restore_unhashable_attributes,
)

# TODO: Add these helper functions to a transform-writing API


def _add_values(v1, v2, function, output_name):
    return function.add_op("Add", inputs=[v1, v2], output_names=[output_name])


def _concat_values(vs, function, dim, output_name):
    return function.add_op(
        "Concat", inputs=vs, attributes={"axis": dim}, output_names=[output_name]
    )


def _identity(v, function, output_name):
    return function.add_op("Identity", inputs=[v], output_names=[output_name])


def _split_value(v, function, num_splits, parallelism_level, dim=0):
    output_names = [f"{v.name}_{parallelism_level}_{i}" for i in range(num_splits)]
    return function.add_op(
        "SplitUniform",
        inputs=[v],
        attributes={"axis": dim, "num_splits": num_splits},
        output_names=output_names,
    )


def _mpi_allgather_values(vs, function, dim, output_names):
    return function.add_op(
        "MPIAllgather",
        inputs=vs,
        attributes={"axis": dim},
        output_names=output_names,
    )


def _mpi_allreduce_values(vs, function, output_names):
    return function.add_op(
        "MPIAllreduce",
        inputs=vs,
        output_names=output_names,
    )


def _mpi_broadcast_value(v, function, devices, parallelism_level):
    output_names = [f"{v.name}_{parallelism_level}_{i}" for i in range(len(devices))]
    return function.add_op(
        "MPIBroadcast",
        inputs=[v],
        attributes={"devices": devices},
        output_names=output_names,
    )


def _mpi_scatter_value(v, function, dim, devices, parallelism_level):
    output_names = [f"{v.name}_{parallelism_level}_{i}" for i in range(len(devices))]
    return function.add_op(
        "MPIScatter",
        inputs=[v],
        attributes={"axis": dim, "devices": devices},
        output_names=output_names,
    )


def _send_value(v, function, device, output_name):
    return function.add_op(
        "Send",
        inputs=[v],
        attributes={"device": device},
        output_names=[output_name],
    )


def _get_op_to_stage_map(stages):
    """Given a list of stages, returns a map from each op in each
    stage to the encompassing stage."""
    op_to_stage = {}
    for stage in stages:
        for op in stage.ops:
            op_to_stage[op] = stage
    return op_to_stage


def _get_consumer_devices_for_pp_value(
    value, function, op_to_stage_map, pp_devices, partition_map
):
    """Returns the set of consumer devices for a pipeline parallel value given
    the corresponding partition map."""
    consumers = function.consumers[value]
    consumer_stages = (op_to_stage_map[op] for op in consumers)
    consumer_devices = set(
        partition_map[consumer_stage] for consumer_stage in consumer_stages
    ).intersection(set(pp_devices))
    return consumer_devices


def _partition_inputs_dp(function, device_tree):
    """Partitions inputs using data parallelism."""

    device_tree_root = tuple(device_tree.keys())[0]
    dp_devices = tuple(sorted(device_tree[device_tree_root].keys()))
    dp_inputs = {}
    if len(dp_devices) > 1:
        # If using data parallelism, partition the inputs and labels
        # and replicate the weights.
        for inp in function.inputs:
            if inp.name == "input1":
                dp_inputs[inp] = _mpi_scatter_value(
                    inp, function, dim=0, devices=dp_devices, parallelism_level="dp"
                )
            else:
                dp_inputs[inp] = _mpi_broadcast_value(
                    inp, function, devices=dp_devices, parallelism_level="dp"
                )
    else:
        # If not using data parallelism, just forward the values from
        # the default device.
        for inp in function.inputs:
            dp_inputs[inp] = [
                _send_value(
                    inp, function, dp_devices[0], output_name=f"{inp.name}_dp_0"
                )
            ]
    return dp_inputs


def _partition_input_hp(inp, function, devices, dim, n_head=None):
    """Partitions the given input using horizontal parallelism.

    Megatron-style parallelism requires splitting the weight matrices
    into 3 parts (Q, K, V) and then dividing each sub-matrix by the
    horizontal parallel degree H. However, this transform must first
    partition the weight matrices by H before doing the split into
    Q, K, V. Therefore to account for the Split op in the graph,
    we must do the following in this initial step:
      1) Split the matrix into Q, K, V
      2) Further split each sub-matrix into H matrices
      3) Re-assemble the full matrix in the form
         Q_1, K_1, V_1, Q_2, K_2, V_2, ..., Q_H, K_H, V_H
      4) Scatter the reassembled matrix to the horizontal parallel devices
    """
    Q, K, V = _split_value(inp, function, 3, parallelism_level="hp", dim=dim)

    hp_degree = len(devices)
    Qs_hp = _split_value(Q, function, hp_degree, parallelism_level="hp", dim=dim)
    Ks_hp = _split_value(K, function, hp_degree, parallelism_level="hp", dim=dim)
    Vs_hp = _split_value(V, function, hp_degree, parallelism_level="hp", dim=dim)

    rearranged_inp = _concat_values(
        tuple(chain.from_iterable(zip(Qs_hp, Ks_hp, Vs_hp))), function, dim, inp.name
    )
    return _mpi_scatter_value(
        rearranged_inp,
        function,
        devices=devices,
        dim=dim,
        parallelism_level="hp",
    )


def _partition_inputs_hp(function, device_tree, dp_inputs):
    """Partitions inputs using horizontal parallelism."""
    device_tree_root = tuple(device_tree.keys())[0]
    dp_devices = tuple(sorted(device_tree[device_tree_root].keys()))
    hp_inputs = {}
    for i, dp_device in enumerate(dp_devices):
        hp_devices = tuple(sorted(device_tree[device_tree_root][dp_device].keys()))
        # If using horizontal parallelism, replicate the inputs and labels
        # and partition the weights. We do this once for each
        # data parallel partition.
        if len(hp_devices) > 1:
            for inp in function.inputs:
                if "c_attn.weight" in inp.name:
                    hp_inputs[dp_inputs[inp][i]] = _partition_input_hp(
                        dp_inputs[inp][i],
                        function,
                        devices=hp_devices,
                        dim=1,
                    )
                elif "c_attn.bias" in inp.name:
                    hp_inputs[dp_inputs[inp][i]] = _partition_input_hp(
                        dp_inputs[inp][i],
                        function,
                        devices=hp_devices,
                        dim=0,
                    )
                elif "c_fc.weight" in inp.name:
                    hp_inputs[dp_inputs[inp][i]] = _mpi_scatter_value(
                        dp_inputs[inp][i],
                        function,
                        devices=hp_devices,
                        dim=1,
                        parallelism_level="hp",
                    )
                elif (
                    "attn.c_proj.weight" in inp.name
                    or "c_fc.bias" in inp.name
                    or "mlp.c_proj.weight" in inp.name
                ):
                    hp_inputs[dp_inputs[inp][i]] = _mpi_scatter_value(
                        dp_inputs[inp][i],
                        function,
                        devices=hp_devices,
                        dim=0,
                        parallelism_level="hp",
                    )
                else:
                    hp_inputs[dp_inputs[inp][i]] = _mpi_broadcast_value(
                        dp_inputs[inp][i],
                        function,
                        devices=hp_devices,
                        parallelism_level="hp",
                    )
        else:
            # If not using horizontal parallelism, no action necessary here.
            for inp in function.inputs:
                hp_inputs[dp_inputs[inp][i]] = [dp_inputs[inp][i]]
    return hp_inputs


def _partition_inputs_pp(
    init_function,
    device_tree,
    dp_inputs,
    hp_inputs,
    num_microbatches,
    function,
    transformed_inputs,
    partition_maps,
    op_to_stage_maps,
):
    """Partitions inputs using pipeline parallelism."""
    device_tree_root = tuple(device_tree.keys())[0]
    dp_devices = tuple(sorted(device_tree[device_tree_root].keys()))
    pp_inputs = defaultdict(dict)
    for i, dp_device in enumerate(dp_devices):
        hp_devices = tuple(sorted(device_tree[device_tree_root][dp_device].keys()))
        for j, hp_device in enumerate(hp_devices):
            pp_devices = device_tree[device_tree_root][dp_device][hp_device]
            for orig_inp in function.inputs:
                inp = transformed_inputs[orig_inp]
                hp_input = hp_inputs[dp_inputs[inp][i]][j]
                if len(pp_devices) > 1:
                    # If using pipeline parallelism, split the input query along the
                    # batch dimension and send all other inputs to their respective devices
                    # according to the partition map. We do this once for every horizontal
                    # parallel partition (and corresponding data parallel partition).
                    if inp.name == "input1":
                        pp_inputs[hp_input][0] = _split_value(
                            hp_input,
                            init_function,
                            num_splits=num_microbatches,
                            parallelism_level="pp",
                        )
                    else:
                        consumer_devices = _get_consumer_devices_for_pp_value(
                            orig_inp,
                            function,
                            op_to_stage_maps[i],
                            pp_devices,
                            partition_maps[i][j],
                        )
                        for consumer_device in consumer_devices:
                            if consumer_device != hp_device:
                                pp_input = _send_value(
                                    hp_input,
                                    init_function,
                                    consumer_device,
                                    output_name=f"{hp_input.name}_pp_all",
                                )
                            else:
                                pp_input = _identity(
                                    hp_input,
                                    init_function,
                                    output_name=f"{hp_input.name}_pp_all",
                                )
                            pp_inputs[hp_input][pp_devices.index(consumer_device)] = [
                                pp_input for _ in range(num_microbatches)
                            ]
                else:
                    # If not using pipeline parallelism, no action necessary here.
                    pp_inputs[hp_input][0] = [hp_input]
    return pp_inputs


def _pipeline_parallel_partition(function, pp_degree, devices):
    """Partitions the function into pipeline parallel stages."""

    # Assemble blocks using MLP Gemm ops as cut points.
    blocks = []
    cur_block = []
    for op in function.ops:
        cur_block.append(op)
        if op.op_type == "Gemm" and any(
            "mlp.c_proj.weight" in inp.name for inp in op.inputs
        ):
            blocks.append(cur_block)
            cur_block = []
    blocks.append(cur_block)
    subfunctions = [
        function.get_subfunction(block, name=f"{function.name} block {i}")
        for i, block in enumerate(blocks)
    ]

    # Places blocks on each device.
    get_roundrobin = roundrobin.basic(list(range(pp_degree)))
    device_order = sorted([get_roundrobin() for _ in range(len(subfunctions))])
    partition_map = {}
    for i in range(len(subfunctions)):
        partition_map[subfunctions[i]] = devices[device_order[i]]
    return partition_map


def _get_device_tree(dp_degree, hp_degree, pp_degree, devices):
    """Constructs a hierarchical device tree given a D/H/P parallelism specification.

    For a list of devices [0, 1, 2, 3, 4, 5, 6, 7, 8] and 2/2/2 D/H/P parallelism,
    the returned device tree will be the following:

    {
      0: {
        1: {
          1: (1, 2),
          3: (3, 4)
        },
        5: {
          5: (5, 6),
          7: (7, 8)
        }
      }
    }

    which represents the following hierarchical topology:

                      0
                   /     \
                 /         \
               /             \
              1               5
           /     \         /     \
          1       3       5       7
         / \     / \     / \     / \
        1   2   3   4   5   6   7   8
    """
    world_size = dp_degree * hp_degree * pp_degree
    dp_size = world_size // dp_degree
    hp_size = dp_size // hp_degree
    device_tree = {
        devices[0]: {
            devices[1 + i * dp_size]: {
                devices[1 + i * dp_size + j * hp_size]: tuple(
                    devices[
                        1
                        + i * dp_size
                        + j * hp_size : 1
                        + i * dp_size
                        + (j + 1) * hp_size
                    ]
                )
                for j in range(hp_degree)
            }
            for i in range(dp_degree)
        }
    }
    return device_tree


def update_attributes(
    op_type,
    attributes,
    attribute_map,
    old_d_embd,
    new_d_embd,
    old_n_head,
    new_n_head,
    new_device=None,
):
    """Updates attributes for Split and Constant ops to reflect new model paramters."""
    if op_type == "Split":
        if "split" in attributes and attributes["split"] == (
            old_d_embd,
            old_d_embd,
            old_d_embd,
        ):
            assert len(attributes) == 2
            attributes = frozendict(
                {
                    "axis": attributes["axis"],
                    "split": (
                        new_d_embd,
                        new_d_embd,
                        new_d_embd,
                    ),
                }
            )
    elif op_type == "Constant":
        value = attribute_map[("value", attributes["value"])]
        if (
            isinstance(value, np.ndarray)
            and value.shape == (1,)
            and value[0] == old_n_head
        ):
            value = np.array([new_n_head])
            sanitized_value = value.tobytes()
            new_device = new_device if new_device is not None else attributes["device"]
            attributes = frozendict({"value": sanitized_value, "device": new_device})
            attribute_map[("value", sanitized_value)] = value
        elif new_device is not None:
            sanitized_value = attributes["value"]
            attributes = frozendict({"value": sanitized_value, "device": new_device})
    return attributes


def gpt2_dhp_transform(
    function,
    dp_degree,
    hp_degree,
    pp_degree,
    devices,
    num_microbatches,
    d_embd,
    n_head,
    debug=False,
):
    """Automatically distributes a GPT-2 function using D/H/P hybrid parallelism."""

    if debug:
        logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

    if pp_degree > 1 and num_microbatches == 1:
        raise ValueError(
            "# of microbatches must be > 1 for pipeline parallel degree > 1"
        )

    # Temporarily remove unhashable attributes.
    (function, attribute_map) = sanitize_unhashable_attributes(function)

    # Initialize the transformed function and construct the device tree given the
    # specified parallelism dimensions.
    fn_name = f"{function.name}_{dp_degree}_{hp_degree}_{pp_degree}_{num_microbatches}"
    transformed_function = FunctionMaker(name=fn_name)
    device_tree = _get_device_tree(dp_degree, hp_degree, pp_degree, devices)
    device_tree_root = tuple(device_tree.keys())[0]
    dp_devices = tuple(sorted(device_tree[device_tree_root].keys()))
    # A list of lists of horizontal parallel devices that synchronize
    # across data parallel partitions.
    hp_device_groups = list(
        zip(
            *[
                tuple(sorted(device_tree[device_tree_root][dp_device].keys()))
                for dp_device in dp_devices
            ]
        )
    )

    # Construct pipeline parallel partitions and schedules for each
    # horizontal parallel partition.
    # A map with the following structure:
    # Data parallel partition ID
    # |-> Attention block (subfunction)
    #     |-> Assigned device
    partition_maps = defaultdict(dict)
    # A list of pipeline parallel schedules, with one schedule
    # (represented as a list of dicts) for every horizontal parallel partition.
    pp_schedules = defaultdict(list)
    op_to_stage_maps = {}
    for i, dp_device in enumerate(device_tree[device_tree_root]):
        hp_devices = tuple(sorted(device_tree[device_tree_root][dp_device].keys()))
        # Construct the pipeline parallel schedules for each horizontal parallel partition.
        for j, hp_device in enumerate(hp_devices):
            pp_devices = device_tree[device_tree_root][dp_device][hp_device]
            partition_maps[i][j] = _pipeline_parallel_partition(
                function, pp_degree, pp_devices
            )
            op_to_stage_maps[i] = _get_op_to_stage_map(partition_maps[i][j].keys())
            scheduler = PipeDreamScheduler(num_microbatches)
            schedule = scheduler.schedule(function, partition_maps[i][j])
            pp_schedules[i].append(schedule)

    # An init function that moves weights/inputs to correct devices.
    init_function = FunctionMaker(name=fn_name + "_init")
    transformed_inputs = {}
    for inp in function.inputs:
        v = init_function.add_input_value(inp.name, inp.type)
        transformed_inputs[inp] = v

    # Partition inputs across each parallelism dimension.
    dp_inputs = _partition_inputs_dp(init_function, device_tree)
    hp_inputs = _partition_inputs_hp(init_function, device_tree, dp_inputs)
    pp_inputs = _partition_inputs_pp(
        init_function,
        device_tree,
        dp_inputs,
        hp_inputs,
        num_microbatches,
        function,
        transformed_inputs,
        partition_maps,
        op_to_stage_maps,
    )
    init_function = init_function.finalize()

    # Inputs of transformed_function are outputs of init_function.
    for v in init_function.outputs:
        transformed_function.inputs.append(v)

    dp_outputs = defaultdict(list)
    for i, dp_device in enumerate(device_tree[device_tree_root]):
        # A map with the following structure:
        # original intermediate value
        # |-> horizontal parallel partition ID
        #     |-> pipeline parallel partition ID
        #         |-> microbatch ID
        #             |-> transformed intermediate value
        intermediate_value_map = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

        # Jointly iterate through all the schedules, timestep by timestep.
        # Timesteps will be a tuple of dicts corresponding to the pipeline parallel
        # schedules at this timestep (represented as a dict) for each horizontal
        # parallel partition. The keys (devices) for each schedule will be different,
        # but the values should be the same. This iteration strategy is necessary
        # for Megatron-style synchronization.
        hp_devices = tuple(sorted(device_tree[device_tree_root][dp_device].keys()))
        for timesteps in zip(*pp_schedules[i]):
            # For a given set of timesteps, iterate through in order of matching
            # horizontal parallel devices.
            for devices in zip(*tuple(sorted(ts.keys()) for ts in timesteps)):
                # Verify that for this group of horizontal parallel devices the
                # corresponding pipeline parallel stage is exactly the same.
                assert (
                    len(set(ts[device] for ts, device in zip(timesteps, devices))) == 1
                )
                assert len(devices) == hp_degree
                stage, microbatch_id = timesteps[0][devices[0]]
                logging.debug(
                    f"Scheduling stage {stage.name}, microbatch {microbatch_id} "
                    f"on device(s) {devices}"
                )
                for op in stage.ops:
                    # Collect inputs for this op.
                    for j, device in enumerate(devices):
                        logging.debug(
                            f"Scheduling op {op} on device {device.device_id}"
                        )
                        pp_devices = device_tree[device_tree_root][dp_device][
                            hp_devices[j]
                        ]
                        k = pp_devices.index(device)
                        input_values = []
                        for inp in op.inputs:
                            # Retrieve the transformed input value from the appropriate
                            # data structure depending on whether the original input is
                            # a function input or an intermediate value.
                            if inp in function.inputs:
                                v = transformed_inputs[inp]
                                dp_v = dp_inputs[v][i]
                                hp_v = hp_inputs[dp_v][j]
                                pp_v = pp_inputs[hp_v][k][microbatch_id]
                                input_values.append(pp_v)
                            else:
                                output_value = intermediate_value_map[j][k][
                                    microbatch_id
                                ][inp]
                                input_values.append(output_value)
                        # Add the op once for each device to the transformed function.
                        if op.op_type == "Split" or op.op_type == "Constant":
                            attributes = update_attributes(
                                op.op_type,
                                op.attributes,
                                attribute_map,
                                old_d_embd=d_embd,
                                new_d_embd=d_embd // hp_degree,
                                old_n_head=n_head,
                                new_n_head=n_head // hp_degree,
                                new_device=device,
                            )
                        else:
                            attributes = op.attributes
                        transformed_outputs = transformed_function.add_op(
                            op.op_type,
                            name=op.name,
                            inputs=input_values,
                            attributes=attributes,
                            output_names=[
                                (
                                    f"{v.name}_dp_{i}_hp_{j}_pp_{microbatch_id}"
                                    f"_device_{device.device_id}"
                                )
                                for v in op.outputs
                            ],
                        )
                        if not isinstance(transformed_outputs, tuple):
                            transformed_outputs = (transformed_outputs,)
                        for output, transformed_output in zip(
                            op.outputs, transformed_outputs
                        ):
                            assert (
                                output
                                not in intermediate_value_map[j][k][microbatch_id]
                            )
                            intermediate_value_map[j][k][microbatch_id][
                                output
                            ] = transformed_output

                    # Reset variables.
                    j = None
                    k = None
                    device = None

                    # Aggregate horizontal parallel outputs.
                    if hp_degree > 1:
                        if op.op_type == "Gemm" and any(
                            [
                                "attn.c_proj.weight" in inp.name
                                or "mlp.c_proj.weight" in inp.name
                                for inp in op.inputs
                            ]
                        ):
                            for output in op.outputs:
                                value_names = tuple(
                                    intermediate_value_map[j][k][microbatch_id][output]
                                    for j in range(len(devices))
                                    for k in intermediate_value_map[j]
                                    if output
                                    in intermediate_value_map[j][k][microbatch_id]
                                )
                                logging.debug(
                                    f"Doing horizontal parallel reduction for "
                                    f"microbatch {microbatch_id} for {value_names}"
                                )
                                aggregated_hp_outputs = []
                                for j, device in enumerate(devices):
                                    pp_devices = device_tree[device_tree_root][
                                        dp_device
                                    ][hp_devices[j]]
                                    aggregated_hp_outputs.append(
                                        intermediate_value_map[j][
                                            pp_devices.index(device)
                                        ][microbatch_id][output]
                                    )
                                reduced_outputs = _mpi_allreduce_values(
                                    tuple(aggregated_hp_outputs),
                                    transformed_function,
                                    output_names=[
                                        (
                                            f"{output.name}_dp_{i}_hp_all_pp_"
                                            f"{microbatch_id}_device_{device.device_id}"
                                        )
                                        for j, device in enumerate(devices)
                                    ],
                                )
                                assert len(reduced_outputs) == len(devices)
                                for j, (device, reduced_output) in enumerate(
                                    zip(devices, reduced_outputs)
                                ):
                                    pp_devices = device_tree[device_tree_root][
                                        dp_device
                                    ][hp_devices[j]]
                                    k = pp_devices.index(device)
                                    intermediate_value_map[j][k][microbatch_id][
                                        output
                                    ] = reduced_output

                    # Aggregate pipeline parallel outputs.
                    for output in op.outputs:
                        if output in function.outputs:
                            for j, device in enumerate(devices):
                                pp_devices = device_tree[device_tree_root][dp_device][
                                    hp_devices[j]
                                ]
                                k = pp_devices.index(device)
                                mb_k_output = intermediate_value_map[j][k][
                                    microbatch_id
                                ][output]
                                match = re.search("hp\_(.*)\_pp", mb_k_output.name)
                                hp_level = match.group(1)
                                if microbatch_id == 0:
                                    # We clone the output from the first microbatch to create
                                    # the aggregated output.
                                    if num_microbatches > 1:
                                        intermediate_value_map[j][k]["all"][
                                            output
                                        ] = _identity(
                                            mb_k_output,
                                            transformed_function,
                                            f"{output.name}_dp_{i}_hp_{hp_level}_pp_all_"
                                            f"device_{device.device_id}",
                                        )
                                    else:
                                        intermediate_value_map[j][k]["all"][
                                            output
                                        ] = mb_k_output

                                else:
                                    # For all subsequent microbatches, we aggregate into the
                                    # specially designated aggregation output. In particular,
                                    # we add weights together and concatenate batch-dependent
                                    # values together.
                                    assert output in intermediate_value_map[j][k]["all"]
                                    mb_all_output = intermediate_value_map[j][k]["all"][
                                        output
                                    ]
                                    assert (
                                        re.search(
                                            "hp\_(.*)\_pp", mb_all_output.name
                                        ).group(1)
                                        == hp_level
                                    )
                                    logging.debug(
                                        f"Doing pipeline parallel aggregation for {mb_all_output} "
                                        f"and {mb_k_output} on device {device.device_id}"
                                    )
                                    intermediate_value_map[j][k]["all"][
                                        output
                                    ] = _concat_values(
                                        (mb_all_output, mb_k_output),
                                        transformed_function,
                                        dim=0,
                                        output_name=(
                                            f"{output.name}_dp_{i}_hp_{hp_level}_"
                                            f"pp_all_device_{device.device_id}"
                                        ),
                                    )

            # Forward any timestep outputs to the next pipeline parallel partition.
            if pp_degree > 1:
                for devices in zip(*tuple(sorted(ts.keys()) for ts in timesteps)):
                    logging.debug(f"Forwarding outputs for stage {stage.name}...")
                    stage, microbatch_id = timesteps[0][devices[0]]
                    for j, device in enumerate(devices):
                        pp_devices = device_tree[device_tree_root][dp_device][
                            hp_devices[j]
                        ]
                        k = pp_devices.index(device)
                        for output in stage.outputs:
                            # An output is forwarded when its consumer devices reside
                            # on a different device than the current stage's device.
                            transformed_output = intermediate_value_map[j][k][
                                microbatch_id
                            ][output]
                            consumer_devices = _get_consumer_devices_for_pp_value(
                                output,
                                function,
                                op_to_stage_maps[i],
                                pp_devices,
                                partition_maps[i][j],
                            )
                            logging.debug(
                                f"Consumer devices for output {output.name}, "
                                f"microbatch {microbatch_id}, "
                                f"device {device.device_id}: "
                                f"{[d.device_id for d in consumer_devices]}"
                            )
                            for consumer_device in consumer_devices:
                                if device != consumer_device:
                                    logging.debug(
                                        f"Sending value {output.name} to "
                                        f"device {consumer_device.device_id}"
                                    )
                                    intermediate_value_map[j][
                                        pp_devices.index(consumer_device)
                                    ][microbatch_id][output] = _send_value(
                                        transformed_output,
                                        transformed_function,
                                        consumer_device,
                                        output_name=(
                                            f"{output.name}_dp_{i}_hp_{j}_pp_"
                                            f"{microbatch_id}_device_"
                                            f"{consumer_device.device_id}"
                                        ),
                                    )

        # Collect the pipeline parallel aggregated function outputs
        # from horizontal parallel partitions to do data parallel aggregation.
        for output in function.outputs:
            dp_outputs[output].append(
                tuple(
                    intermediate_value_map[j][k]["all"][output]
                    for j in intermediate_value_map
                    for k in intermediate_value_map[j]
                    if output in intermediate_value_map[j][k]["all"]
                )
            )
            # There should only be as many pipeline parallel aggregated function outputs
            # as there are horizontal parallel partitions.
            assert len(dp_outputs[output][-1]) == len(hp_devices)

    # Aggregate data parallel outputs.
    if dp_degree > 1:
        for output in dp_outputs:
            logging.debug(f"Doing data parallel reduction for {dp_outputs[output]}")
            hp_groups = list(zip(*dp_outputs[output]))
            if output.name == "output1":
                for i, hp_group in enumerate(hp_groups):
                    if hp_degree > 1:
                        hp_device_group_str = ",".join(
                            [str(d.device_id) for d in hp_device_groups[i]]
                        )
                    else:
                        hp_device_group_str = "all"
                    _mpi_allgather_values(
                        hp_group,
                        transformed_function,
                        dim=0,
                        output_names=[
                            f"{output.name}_dp_all_hp_{hp_device_group_str}_pp_all"
                            for _ in range(len(hp_group))
                        ],
                    )
            else:
                # Do nothing for other outputs
                pass

    # Hack to get around unhashable numpy array attributes
    # TODO: Fix this more gracefully?
    transformed_function = restore_unhashable_attributes(
        transformed_function, attribute_map
    )

    return init_function, transformed_function.finalize()
