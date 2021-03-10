from collections import defaultdict
import numpy as np
import pdb
import re

import dist_ir
from dist_ir.importer import import_from_onnx, parse_tensor_from_file
from dist_ir.ir import FunctionMaker, cpprint, pformat, Device, Topology, Value
from dist_ir.executor import infer_types, SequentialExecutor
from dist_ir.executor.cost_model import CostModel
from dist_ir.ir.type import Bool, Float, Int64, Tensor
from dist_ir.transforms import (
    parallel_transform_3d,
    steady_state_transform,
    PipeDreamScheduler,
)

BATCH_SIZE = 64
INPUT_DIM = 64
DGX_BANDWIDTH_GBPS = 200


def mlp(batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, device):
    function = FunctionMaker(name="mlp")
    x = function.add_input_value(
        "x",
        Tensor(dtype=Float(), shape=(batch_size, input_dim), device=device),
    )
    z = function.add_input_value(
        "z",
        Tensor(dtype=Float(), shape=(batch_size, output_dim), device=device),
    )
    weights = []
    input_dim = input_dim
    hidden_dim = hidden_dim
    for i in range(num_hidden_layers - 1):
        w = function.add_input_value(
            f"w{chr(ord('A')+i)}",
            Tensor(dtype=Float(), shape=(input_dim, hidden_dim), device=device),
        )
        input_dim = hidden_dim
        weights.append(w)
    w = function.add_input_value(
        f"w{chr(ord('A')+i+1)}",
        Tensor(dtype=Float(), shape=(hidden_dim, output_dim), device=device),
    )
    weights.append(w)

    a = x
    for i, weight in enumerate(weights):
        y = function.add_op("MatMul", inputs=[a, weight], output_names=[f"y{i}"])
        a = function.add_op("Relu", inputs=[y], output_names=[f"a{i}"])

    l = function.add_op(
        "Loss", inputs=[a, z], attributes={"N": batch_size}, output_names=["l"]
    )
    dl = function.add_op(
        "LossGrad",
        inputs=[a, z],
        attributes={"N": batch_size},
        output_names=["dl"],
    )

    dy = dl
    for i, weight in enumerate(weights[::-1]):
        i = len(weights) - i - 1
        da = function.add_op(
            "ReluGrad",
            inputs=[function.ops[2 * i + 1].inputs[0], dy],
            output_names=[f"da{i}"],
        )
        dy, dw = function.add_op(
            "MatMulGrad",
            inputs=[function.ops[2 * i].inputs[0], weights[i], da],
            output_names=[f"dy{i}", f"dw{chr(ord('A')+i)}"],
        )
    return function.finalize()


def add_devices_to_topology(topology, num_devices):
    for i in range(num_devices):
        topology.add_device("gpu")
    devices = topology.devices
    for i in range(0, len(devices)):
        for j in range(i + 1, len(devices)):
            topology.set_bandwidth(devices[i], devices[j], DGX_BANDWIDTH_GBPS)
    return topology


def _verify_no_hp(outputs, transformed_outputs, dp=False):
    for output, transformed_output in zip(outputs, transformed_outputs):
        if dp:
            np.testing.assert_array_almost_equal(output, transformed_output[0])
        else:
            np.testing.assert_array_almost_equal(output, transformed_output)


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
        if dp:
            aggregated_outputs[key].append(v[0])
        else:
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
            aggregated_output,
            indexed_outputs[output_name],
            decimal=3
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
    topology = Topology()
    d0 = topology.add_device("gpu")
    function = mlp(batch_size, input_dim, input_dim, input_dim, num_hidden_layers, d0)
    function = infer_types(function, function.inputs)
    world_size = dp_degree * hp_degree * pp_degree
    add_devices_to_topology(topology, world_size)

    transformed_function = parallel_transform_3d(
        function,
        dp_degree,
        hp_degree,
        pp_degree,
        topology.devices,
        num_microbatches,
    )
    transformed_function = infer_types(
        transformed_function, transformed_function.inputs
    )

    input_data = [np.random.normal(size=inp.type.shape) for inp in function.inputs]
    ex = SequentialExecutor("numpy")
    outputs = ex.compute(function, input_data)
    transformed_outputs = ex.compute(transformed_function, input_data)

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
    test_dp_hp_pp()
