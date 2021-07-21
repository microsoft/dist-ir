from collections import defaultdict
import numpy as np
import pytest
import re

from dist_ir.importer import import_from_onnx, parse_tensor_from_file
from dist_ir.ir import FunctionMaker, cpprint, pformat, Device, Topology, Value
from dist_ir.executor import infer_types, SequentialExecutor
from dist_ir.executor.cost_model import CostModel
from dist_ir.ir.type import Bool, Float32, Int64, Tensor
from dist_ir.transforms import (
    mlp_dhp_transform,
    PipeDreamScheduler,
)
from examples import mlp

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


def _test_helper(
    batch_size=BATCH_SIZE,
    num_hidden_layers=8,
    input_dim=INPUT_DIM,
    dp_degree=1,
    hp_degree=1,
    pp_degree=1,
    num_microbatches=1,
):
    world_size = dp_degree * hp_degree * pp_degree
    topology = mlp.get_topology(world_size)
    function = mlp.mlp(
        batch_size,
        input_dim,
        input_dim,
        input_dim,
        num_hidden_layers,
        topology.devices[0],
    )
    function = infer_types(function, function.inputs)

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

    input_data = [np.random.normal(size=inp.type.shape) for inp in function.inputs]
    ex = SequentialExecutor("numpy")
    outputs = ex.compute(function, input_data)
    dist_input_data = ex.compute(init_function, input_data)
    transformed_outputs = ex.compute(transformed_function, dist_input_data)

    if hp_degree > 1:
        _verify_hp(
            function, transformed_function, outputs, transformed_outputs, dp_degree > 1
        )
    else:
        _verify_no_hp(outputs, transformed_outputs, dp_degree > 1)


def test_dp_only():
    _test_helper(dp_degree=2)


def test_hp_only():
    _test_helper(hp_degree=2)


def test_pp_only():
    _test_helper(pp_degree=2, num_microbatches=2)


def test_dp_hp():
    _test_helper(dp_degree=2, hp_degree=2)


def test_dp_pp():
    _test_helper(dp_degree=2, pp_degree=2, num_microbatches=2)


def test_hp_pp():
    _test_helper(hp_degree=2, pp_degree=2, num_microbatches=2)


def test_dp_hp_pp():
    _test_helper(dp_degree=2, hp_degree=2, pp_degree=2, num_microbatches=2)


if __name__ == "__main__":
    test_dp_only()
    test_hp_only()
    test_pp_only()
    test_dp_hp()
    test_hp_pp()
    test_dp_hp_pp()
