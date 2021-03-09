from collections import OrderedDict
import numpy as np

from dist_ir.ir import Device, FunctionMaker, cpprint
from dist_ir.ir.type import Float, Tensor
from dist_ir.transforms import hybrid_transform
from dist_ir.executor import SequentialExecutor


def test_hybrid_transform():
    function = FunctionMaker()
    num_microbatches = 2
    batch_size = 128
    input_dim = 16
    hidden_dim = 32
    output_dim = 64
    world_size = 8
    devices = [Device(i, "gpu") for i in range(world_size + 1)]

    x = function.add_input_value(
        "x", Tensor(dtype=Float(), shape=(batch_size, input_dim), device=devices[0])
    )
    wA = function.add_input_value(
        "wA", Tensor(dtype=Float(), shape=(input_dim, hidden_dim), device=devices[0])
    )
    wB = function.add_input_value(
        "wB", Tensor(dtype=Float(), shape=(hidden_dim, input_dim), device=devices[0])
    )
    a = function.add_op(op_type="MatMul", inputs=[x, wA], output_names=["a"])
    y = function.add_op(op_type="MatMul", inputs=[a, wB], output_names=["y"])

    function = function.finalize()
    dp_config = {
        "input_dims": {function.inputs[0]: 0},
        "reduction_params": {
            function.outputs[0]: {"op_type": "MPIGather", "dim": 0, "device": devices[0]}
        },
        "devices": (devices[1], devices[5]),
        "verify_fn": None,
    }
    hp_config = {
        "input_dims": {function.inputs[1]: 1, function.inputs[2]: 0},
        "reduction_params": {function.outputs[0]: {"op_type": "MPIAllreduce"}},
        "devices": (devices[3], devices[7]),
        "verify_fn": None,
    }
    stages = [
        function.get_subfunction([function.ops[0]], name="f0"),
        function.get_subfunction([function.ops[1]], name="f1"),
    ]
    partition_map = OrderedDict([(stages[0], devices[2]), (stages[1], devices[6])])
    schedule = [
        {devices[2]: (stages[0], 0)},
        {devices[2]: (stages[0], 1), devices[6]: (stages[1], 0)},
        {devices[6]: (stages[1], 1)},
    ]
    pp_config = {
        "num_microbatches": num_microbatches,
        "batch_dims": {function.inputs[0]: 0},
        "reduction_params": {function.outputs[0]: {"op_type": "Concat", "dim": 0}},
        "partition_map": partition_map,
        "schedule": schedule,
    }
    function = hybrid_transform(function, dp_config, hp_config, pp_config)
    cpprint(function)


if __name__ == "__main__":
    test_hybrid_transform()
