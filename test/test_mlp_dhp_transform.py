# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
import itertools
import numpy as np
import pytest
import re

from examples import mlp
from dist_ir.ir import FunctionMaker, get_uniform_topology
from dist_ir.executor import infer_types, sequentially_execute, ConcreteValue
from dist_ir.ir.type import Float32, Tensor
from dist_ir.transforms import mlp_dhp_transform

BATCH_SIZE = 64
INPUT_DIM = 64
DGX_BANDWIDTH_GBPS = 200

np.random.seed(42)


def _verify_no_hp(outputs, transformed_outputs, dp=False):
    for i in range(len(outputs)):
        if not dp:
            j = i
        else:
            j = 2 * i
        np.testing.assert_array_almost_equal(outputs[i], transformed_outputs[j])


def _verify_hp(function, transformed_function, outputs, transformed_outputs, dp=False):
    indexed_outputs = dict(
        list(zip([output.name for output in function.outputs], outputs))
    )
    aggregated_outputs = defaultdict(list)
    for output, v in zip(transformed_function.outputs, transformed_outputs):
        device_suffix = "_device_(.*)" if "device" in output.name else ""
        match = re.search(f"(.*)_dp_(.*)_hp_(.*)_pp_(.*){device_suffix}", output.name)
        assert match is not None
        key = (match.group(1), match.group(2), match.group(4))
        aggregated_outputs[key].append(v)
    for key in aggregated_outputs:
        output_name = key[0]
        if "dw" in output_name:
            weight_id = output_name[2:]
            axis = 1 - ((ord(weight_id) - ord("A")) % 2)
            aggregated_output = np.concatenate(aggregated_outputs[key], axis=axis)
        else:
            aggregated_output = aggregated_outputs[key][0]
        np.testing.assert_array_almost_equal(
            aggregated_output, indexed_outputs[output_name], decimal=3
        )


@pytest.mark.parametrize(
    ("dp_degree", "hp_degree", "pp_degree"),
    list(itertools.product([1, 2], [1, 2], [1, 2])),
)
def test_mlp_dhp_transform(
    dp_degree,
    hp_degree,
    pp_degree,
    batch_size=BATCH_SIZE,
    num_hidden_layers=8,
    input_dim=INPUT_DIM,
):
    num_microbatches = pp_degree
    world_size = dp_degree * hp_degree * pp_degree
    topology = get_uniform_topology(world_size)
    function = mlp.mlp(
        input_dim,
        input_dim,
        input_dim,
        num_hidden_layers,
        topology.devices[0],
    )
    typed_inputs = mlp.get_typed_input_values(
        function.inputs, batch_size, input_dim, input_dim
    )
    function = infer_types(function, typed_inputs)

    init_function, transformed_function = mlp_dhp_transform(
        function,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
        topology.devices,
    )
    init_function = infer_types(init_function, init_function.inputs)
    # init_function.outputs = transformed_function.inputs, so get types from there:
    transformed_function = infer_types(transformed_function, init_function.outputs)
    transformed_function = mlp.add_optimizer_ops(transformed_function)

    input_data = [
        ConcreteValue(
            np.random.normal(size=inp.type.shape) if i != 2 else batch_size,
            topology.devices[0],
        )
        for i, inp in enumerate(typed_inputs)
    ]
    outputs = sequentially_execute(function, input_data)
    dist_input_data = sequentially_execute(init_function, input_data)
    transformed_outputs = sequentially_execute(transformed_function, dist_input_data)

    outputs = [v.val for v in outputs]
    # Verify that transformed_outputs are on expected devices
    assert all(
        o.type.device == v.device
        for o, v in zip(transformed_function.outputs, transformed_outputs)
    )
    transformed_outputs = [v.val for v in transformed_outputs]

    if hp_degree > 1:
        _verify_hp(
            function, transformed_function, outputs, transformed_outputs, dp_degree > 1
        )
    else:
        _verify_no_hp(outputs, transformed_outputs, dp_degree > 1)
