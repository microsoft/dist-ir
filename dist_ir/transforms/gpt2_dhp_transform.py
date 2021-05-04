from collections import defaultdict, Hashable
import math
import logging
import re

from ..ir import cpprint, Op
from ..ir.function import Function, FunctionMaker
from .pipedream_scheduler import PipeDreamScheduler
from .sanitize_attributes_transform import (
    sanitize_unhashable_attributes,
    restore_unhashable_attributes,
)


def _add_values(v1, v2, function, output_name):
    return function.add_op("Add", inputs=[v1, v2], output_names=[output_name])


def _concat_values(v1, v2, function, dim, output_name):
    return function.add_op(
        "Concat", inputs=[v1, v2], attributes={"axis": dim}, output_names=[output_name]
    )


def _identity(v, function, output_name):
    return function.add_op("Identity", inputs=[v], output_names=[output_name])


def _split_value(v, function, num_splits, parallelism_level):
    assert parallelism_level == "pp"
    output_names = [f"{v.name}_{parallelism_level}_{i}" for i in range(num_splits)]
    return function.add_op(
        "SplitDistIR",
        inputs=[v],
        attributes={"dim": 0, "num_splits": num_splits},
        output_names=output_names,
    )


def _mpi_allgather_values(vs, function, dim, output_names):
    return function.add_op(
        "MPIAllgather",
        inputs=vs,
        attributes={"dim": dim},
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
        attributes={"dim": dim, "devices": devices},
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
            # TODO: Partition weights for GPT-2
            for inp in function.inputs:
                if "c_attn.weight" in inp.name or "c_fc.weight" in inp.name:
                    hp_inputs[dp_inputs[inp][i]] = _mpi_scatter_value(
                        dp_inputs[inp][i],
                        function,
                        devices=hp_devices,
                        dim=1,
                        parallelism_level="hp",
                    )
                elif (
                    "c_attn.bias" in inp.name
                    or "attn.c_proj.weight" in inp.name
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
    function,
    device_tree,
    dp_inputs,
    hp_inputs,
    num_microbatches,
):
    """Partitions inputs using pipeline parallelism."""
    device_tree_root = tuple(device_tree.keys())[0]
    dp_devices = tuple(sorted(device_tree[device_tree_root].keys()))
    pp_inputs = {}
    for i, dp_device in enumerate(dp_devices):
        hp_devices = tuple(sorted(device_tree[device_tree_root][dp_device].keys()))
        for j, hp_device in enumerate(hp_devices):
            pp_devices = device_tree[device_tree_root][dp_device][hp_device]
            for inp in function.inputs:
                hp_input = hp_inputs[dp_inputs[inp][i]][j]
                if len(pp_devices) > 1 and inp.name == "input1":
                    # If using pipeline parallelism, split the input along the
                    # batch dimension. No action is necessary for the weights. We do this
                    # once for every horizontal parallel partition (and corresponding data
                    # parallel partition).
                    pp_inputs[hp_input] = _split_value(
                        hp_input,
                        function,
                        num_splits=num_microbatches,
                        parallelism_level="pp",
                    )
                else:
                    # If not using pipeline parallelism, no action necessary here.
                    pp_inputs[hp_input] = [hp_input]
    return pp_inputs


def _get_producers(function):
    producers = {}
    for op in function.ops:
        for output in op.outputs:
            producers[output] = op
    return producers


def _get_subgraph_from_sink(producers, output):
    subgraph = set()
    queue = [producers[output]]
    while len(queue) > 0:
        cur = queue.pop(0)
        subgraph.add(cur)
        for inp in cur.inputs:
            if inp in producers:
                producer = producers[inp]
                if producer not in subgraph:
                    queue.append(producer)
    return subgraph


def _filter_extra_outputs(function):
    # Map from op to set of function output values.
    sinks = defaultdict(set)

    # Map from output value to producer op.
    producers = _get_producers(function)

    # Set the sink for each output producer op to be the output.
    for output in function.outputs:
        producer = producers[output]
        sinks[producer] = set([output])

    # Incrementally propogate the set of sinks for each op by iterating through
    # all ops in reverse topological order.
    ops = list(function.ops)[::-1]
    while len(ops) > 0:
        op = ops.pop(0)
        for output in op.outputs:
            for consumer in function.consumers[output]:
                sinks[op] = sinks[op].union(sinks[consumer])

    # Filter out ops with no sinks other than output1.
    filtered_ops = set()
    for op in sinks:
        if function.outputs[-1] not in sinks[op]:
            filtered_ops.add(op)
    filtered_function = FunctionMaker(name=function.name)
    value_map = {}
    for inp in function.inputs:
        v = filtered_function.add_input_value(inp.name, inp.type)
        value_map[inp] = v
    for op in function.ops:
        if op in filtered_ops:
            continue
        inputs = tuple(value_map[inp] for inp in op.inputs)
        new_op = Op(
            name=op.name,
            op_type=op.op_type,
            inputs=inputs,
            attributes=op.attributes,
            subfunctions=op.subfunctions,
            output_names=tuple(output.name for output in op.outputs),
            output_types=tuple(output.type for output in op.outputs),
        )
        filtered_function.ops.append(new_op)
        for orig_output, new_output in zip(op.outputs, new_op.outputs):
            value_map[orig_output] = new_output
    return filtered_function.finalize()


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
    num_blocks_per_device = len(subfunctions) // pp_degree
    partition_map = {}
    for i in range(len(subfunctions)):
        partition_map[subfunctions[i]] = devices[
            min(i // num_blocks_per_device, len(devices) - 1)
        ]

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


def gpt2_dhp_transform(
    function, dp_degree, hp_degree, pp_degree, devices, num_microbatches
):
    """Automatically distributes a GPT-2 function using D/H/P hybrid parallelism."""

    # Hack to get around unhashable numpy array attributes
    # TODO: Fix this more gracefully?
    orig_function = function
    (function, attribute_map) = sanitize_unhashable_attributes(function)

    function = _filter_extra_outputs(function)

    transformed_function = FunctionMaker(name=function.name)
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

    dp_outputs = defaultdict(list)
    for i, dp_device in enumerate(device_tree[device_tree_root]):
        # pp_schedules is a list of pipeline parallel schedules, with one schedule
        # (represented as a list of dicts) list for every horizontal parallel partition.
        partition_maps = {}
        pp_schedules = []
        hp_devices = tuple(sorted(device_tree[device_tree_root][dp_device].keys()))
        # Construct the pipeline parallel schedules for each horizontal parallel partition.
        for j, hp_device in enumerate(hp_devices):
            pp_devices = device_tree[device_tree_root][dp_device][hp_device]
            partition_maps[j] = _pipeline_parallel_partition(
                function, pp_degree, pp_devices
            )
            op_to_stage_map = _get_op_to_stage_map(partition_maps[j].keys())
            scheduler = PipeDreamScheduler(num_microbatches)
            schedule = scheduler.schedule(function, partition_maps[j])
            pp_schedules.append(schedule)

        # A map from original value to transformed value. Keeps track of values
        # forwarded between pipeline parallel stages on separate devices.
        forwarded_value_map = {}

        # A map with the following structure:
        # original intermediate value
        # |-> horizontal parallel partition ID
        #     |-> microbatch ID
        #         |-> transformed intermediate value
        intermediate_value_map = defaultdict(lambda: defaultdict(dict))

        # A map from microbatch ID to MatMul count. The count is incremented each time
        # a MatMul or MatMulGrad op is executed. Horizontal parallel synchronization
        # is performed when the count reaches an even value.
        matmul_counter = defaultdict(lambda: 0)

        # Jointly iterate through all the schedules, timestep by timestep.
        # Timesteps will be a tuple of dicts corresponding to the schedules
        # at this timestep (represented as a dict) for each horizontal parallel
        # partition. The keys (devices) for each schedule will be different,
        # but the values should be the same. This iteration strategy is necessary
        # for Megatron-style synchronization.
        for timesteps in zip(*pp_schedules):
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
                for op in stage.ops:
                    # Collect inputs for this op.
                    for j, device in enumerate(devices):
                        input_values = []
                        input_devices = []
                        pp_devices = device_tree[device_tree_root][dp_device][
                            hp_devices[j]
                        ]
                        for inp in op.inputs:
                            # Retrieve the transformed input value from the appropriate
                            # data structure depending on whether the original input is
                            # a function input or an intermediate value.
                            if inp in function.inputs:
                                v = transformed_inputs[inp]
                                dp_v = dp_inputs[v][i]
                                hp_v = hp_inputs[dp_v][j]
                                if inp.name == "input1":
                                    pp_v = pp_inputs[hp_v][microbatch_id]
                                else:
                                    pp_v = pp_inputs[hp_v][0]
                                input_values.append(pp_v)
                                input_devices.append(pp_devices[0])
                            else:
                                output_value, output_device = intermediate_value_map[j][
                                    microbatch_id
                                ][inp]
                                input_values.append(output_value)
                                input_devices.append(output_device)
                        # Forward any input values not on the correct device.
                        for idx, (inp, v, d) in enumerate(
                            zip(op.inputs, input_values, input_devices)
                        ):
                            if d != device:
                                if (v, device) in forwarded_value_map:
                                    logging.debug(
                                        f"Found ({v.name}, {device.device_id})"
                                        f"in sent value cache"
                                    )
                                else:
                                    logging.debug(
                                        f"Sending value {inp.name} to"
                                        f"device {device.device_id}"
                                    )
                                    forwarded_value_map[(v, device)] = _send_value(
                                        v,
                                        transformed_function,
                                        device,
                                        output_name=(
                                            f"{inp.name}_dp_{i}_hp_{j}_pp_{microbatch_id}"
                                            f"_device_{device.device_id}"
                                        ),
                                    )
                                input_values[idx] = forwarded_value_map[(v, device)]
                        # Add the op once for each device to the transformed function.
                        attributes = op.attributes
                        if op.op_type == "Split":
                            if "split" in attributes and attributes["split"] == (
                                768,
                                768,
                                768,
                            ):
                                assert len(attributes) == 2
                                new_dim = 768 // hp_degree
                                attributes = {
                                    "axis": attributes["axis"],
                                    "split": (new_dim, new_dim, new_dim),
                                }

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
                                output not in intermediate_value_map[j][microbatch_id]
                            )
                            intermediate_value_map[j][microbatch_id][output] = (
                                transformed_output,
                                device,
                            )

                    # Reset variables.
                    j = None
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
                                    intermediate_value_map[j][microbatch_id][output][0]
                                    for j in range(len(devices))
                                )
                                logging.debug(
                                    f"Doing horizontal parallel reduction for "
                                    f"microbatch {microbatch_id} for {value_names}"
                                )
                                reduced_outputs = _mpi_allreduce_values(
                                    tuple(
                                        intermediate_value_map[j][microbatch_id][
                                            output
                                        ][0]
                                        for j in range(len(devices))
                                    ),
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
                                for k, (d, reduced_output) in enumerate(
                                    zip(devices, reduced_outputs)
                                ):
                                    intermediate_value_map[k][microbatch_id][output] = (
                                        reduced_output,
                                        d,
                                    )

                    # Aggregate pipeline parallel outputs.
                    for output in op.outputs:
                        if output in function.outputs:
                            for j, device in enumerate(devices):
                                mb_k_output, mb_k_device = intermediate_value_map[j][
                                    microbatch_id
                                ][output]
                                assert mb_k_device == device
                                match = re.search("hp\_(.*)\_pp", mb_k_output.name)
                                hp_level = match.group(1)
                                if microbatch_id == 0:
                                    # We clone the output from the first microbatch to create
                                    # the aggregated output.
                                    if num_microbatches > 1:
                                        intermediate_value_map[j]["all"][output] = (
                                            _identity(
                                                mb_k_output,
                                                transformed_function,
                                                f"{output.name}_dp_{i}_hp_{hp_level}_pp_all_"
                                                f"device_{mb_k_device.device_id}",
                                            ),
                                            mb_k_device,
                                        )
                                    else:
                                        intermediate_value_map[j]["all"][output] = (
                                            mb_k_output,
                                            mb_k_device,
                                        )
                                else:
                                    # For all subsequent microbatches, we aggregate into the
                                    # specially designated aggregation output. In particular,
                                    # we add weights together and concatenate batch-dependent
                                    # values together.
                                    assert output in intermediate_value_map[j]["all"]
                                    (
                                        mb_all_output,
                                        mb_all_device,
                                    ) = intermediate_value_map[j]["all"][output]
                                    assert mb_all_device == device
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
                                    intermediate_value_map[j]["all"][output] = (
                                        _concat_values(
                                            mb_all_output,
                                            mb_k_output,
                                            transformed_function,
                                            dim=0,
                                            output_name=(
                                                f"{output.name}_dp_{i}_hp_{hp_level}_"
                                                f"pp_all_device_{mb_all_device.device_id}"
                                            ),
                                        ),
                                        mb_all_device,
                                    )

            # Forward any timestep outputs to the next pipeline parallel partition.
            if pp_degree > 1:
                for devices in zip(*tuple(sorted(ts.keys()) for ts in timesteps)):
                    stage, microbatch_id = timesteps[0][devices[0]]
                    for j, device in enumerate(devices):
                        pp_devices = device_tree[device_tree_root][dp_device][
                            hp_devices[j]
                        ]
                        for output in stage.outputs:
                            # An output is forwarded when its consumer devices reside
                            # on a different device than the current stage's device.
                            transformed_output, d = intermediate_value_map[j][
                                microbatch_id
                            ][output]
                            assert device == d
                            consumers = function.consumers[output]
                            consumer_stages = (op_to_stage_map[op] for op in consumers)
                            consumer_devices = set(
                                partition_maps[j][consumer_stage]
                                for consumer_stage in consumer_stages
                            ).intersection(set(pp_devices))
                            for consumer_device in consumer_devices:
                                if device != consumer_device:
                                    logging.debug(
                                        f"Sending value {output.name} to "
                                        f"device {consumer_device.device_id}"
                                    )

                                    forwarded_value_map[
                                        (transformed_output, consumer_device)
                                    ] = _send_value(
                                        transformed_output,
                                        transformed_function,
                                        consumer_device,
                                        output_name=(
                                            f"{output.name}_dp_{i}_hp_{j}_pp_"
                                            f"{microbatch_id}_device_"
                                            f"{consumer_device.device_id}"
                                        ),
                                    )
        # Collect the pipeline-parallel aggregated function outputs
        # from horizontal parallel partitions to do data parallel aggregation.
        for output in function.outputs:
            dp_outputs[output].append(
                tuple(
                    intermediate_value_map[j]["all"][output][0]
                    for j in intermediate_value_map
                )
            )

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
                """
                for i, hp_group in enumerate(hp_groups):
                    _mpi_allgather_values(
                        hp_group,
                        transformed_function,
                        dim=0,
                        output_names=[f"{output.name}_dp_all_hp_all_pp_all" for _ in range(len(hp_group))],
                    )
                """

    # Hack to get around unhashable numpy array attributes
    # TODO: Fix this more gracefully?
    transformed_function = restore_unhashable_attributes(
        transformed_function, attribute_map
    )

    return transformed_function.finalize()
