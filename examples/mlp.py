import argparse
from collections import defaultdict, OrderedDict
import numpy as np
import onnx
from pathlib import Path
from contextlib import redirect_stdout

import dist_ir
from dist_ir.importer import import_from_onnx, parse_tensor_from_file
from dist_ir.ir import FunctionMaker, cpprint, pformat, Device, Topology, Value
from dist_ir.executor import infer_types, SequentialExecutor, Simulator
from dist_ir.executor.cost_model import CostModel
from dist_ir.ir.type import Bool, Float, Int64, Tensor
from dist_ir.transforms import hybrid_transform_unrolled


def mlp():
    function = FunctionMaker(name="mlp")
    num_microbatches = 2
    batch_size = 128
    input_dim = 16
    hidden_dim = 32
    output_dim = 64
    world_size = 4
    device_speeds = {"gpu": 1.0e13}
    topology = Topology()
    devices = [topology.add_device("gpu") for i in range(world_size + 1)]

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
            function.outputs[0]: {
                "op_type": "MPIGather",
                "dim": 0,
                "device": devices[0],
            }
        },
        "devices": (devices[1], devices[3]),
        "verify_fn": None,
    }
    hp_config = {
        "input_dims": {function.inputs[1]: 1, function.inputs[2]: 0},
        "reduction_params": {
            function.outputs[0]: {"op_type": "MPIReduce", "device": devices[3]}
        },
        "devices": {
            devices[1]: (devices[1], devices[2]),
            devices[3]: (devices[3], devices[4]),
        },
        "verify_fn": None,
    }
    """
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
    """
    function = hybrid_transform_unrolled(function, dp_config, hp_config, pp_config=None)
    cpprint(function, width=250)

    function = infer_types(function, function.inputs)
    cpprint(function, width=250)

    simulator = Simulator(CostModel(topology, device_speeds))
    simulation = simulator.interpret(function, (v.type for v in function.inputs))


if __name__ == "__main__":
    mlp()
