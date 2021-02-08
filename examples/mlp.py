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


def mlp(args, devices):
    function = FunctionMaker(name="mlp")
    x = function.add_input_value(
        "x",
        Tensor(
            dtype=Float(), shape=(args.batch_size, args.input_dim), device=devices[0]
        ),
    )
    z = function.add_input_value(
        "z",
        Tensor(
            dtype=Float(), shape=(args.batch_size, args.output_dim), device=devices[0]
        ),
    )
    weights = []
    input_dim = args.input_dim
    for i, hidden_dim in enumerate(args.hidden_dims):
        w = function.add_input_value(
            f"w{chr(ord('A')+i)}",
            Tensor(dtype=Float(), shape=(input_dim, hidden_dim), device=devices[0]),
        )
        input_dim = hidden_dim
        weights.append(w)
    w = function.add_input_value(
        f"w{chr(ord('A')+i+1)}",
        Tensor(dtype=Float(), shape=(input_dim, args.output_dim), device=devices[0]),
    )
    weights.append(w)

    y = x
    for i, weight in enumerate(weights):
        y = function.add_op("MatMul", inputs=[y, weight], output_names=[f"y{i}"])
        # TODO: Add activation function?

    l = function.add_op(
        "Loss", inputs=[y, z], attributes={"N": args.batch_size}, output_names=["l"]
    )
    dl = function.add_op(
        "LossGrad",
        inputs=[y, z],
        attributes={"N": args.batch_size},
        output_names=["dl"],
    )

    dy = dl
    for i, weight in enumerate(weights[::-1]):
        i = len(weights) - i - 1
        dy, dw = function.add_op(
            "MatMulGrad",
            inputs=[function.ops[i].outputs[0], weights[i], dy],
            output_names=[f"dy{i}", f"dw{chr(ord('A')+i)}"],
        )

    return function.finalize()


def main(args):
    device_speeds = {"gpu": 1.0e13}
    topology = Topology()
    devices = [topology.add_device("gpu") for i in range(args.world_size + 1)]

    function = mlp(args, topology.devices)
    cpprint(function)
    function = infer_types(function, function.inputs)
    cpprint(function)
    ex = SequentialExecutor("numpy")
    input_data = [np.random.normal(size=(inp.type.shape)) for inp in function.inputs]
    res = ex.compute(function, input_data)

    dp_config = {
        "input_dims": {function.inputs[0]: 0, function.inputs[1]: 0},
        "reduction_params": {
            function.outputs[0]: {
                "op_type": "MPIReduce",
                "device": devices[0],
            },
            function.outputs[1]: {
                "op_type": "MPIReduce",
                "device": devices[0],
            },
            function.outputs[2]: {
                "op_type": "MPIGather",
                "dim": 0,
                "device": devices[0],
            },
            function.outputs[3]: {
                "op_type": "MPIReduce",
                "device": devices[0],
            },
        },
        "devices": (devices[1], devices[3]),
        "verify_fn": None,
    }
    hp_config = {
        "input_dims": {function.inputs[2]: 1, function.inputs[3]: 0},
        "reduction_params": {
            function.outputs[0]: {
                "op_type": "MPIReduce",
            },
            function.outputs[1]: {
                "op_type": "MPIGather",
                "dim": 1,
            },
            function.outputs[2]: {
                "op_type": "MPIGather",
                "dim": 0,
            },
            function.outputs[3]: {
                "op_type": "MPIReduce",
                "dim": 0,
            },
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

    transformed_function = hybrid_transform_unrolled(
        function, dp_config, hp_config, pp_config=None
    )
    cpprint(transformed_function)
    transformed_function = infer_types(
        transformed_function, transformed_function.inputs
    )
    cpprint(transformed_function, width=250)
    transformed_res = ex.compute(function, input_data)
    for a, b in zip(res, transformed_res):
        np.testing.assert_array_almost_equal(a, b)

    simulator = Simulator(CostModel(topology, device_speeds))
    simulation = simulator.interpret(
        transformed_function, (v.type for v in transformed_function.inputs)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLP example")
    parser.add_argument(
        "--num_microbatches",
        type=int,
        default=2,
        help="Number of pipeline parallel microbatches per minibatch",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--input_dim", type=int, default=16, help="Input data dimension"
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[32],
        help="Hidden layer dimensions",
    )
    parser.add_argument("--output_dim", type=int, default=64, help="Output dimension")
    parser.add_argument("--world_size", type=int, default=4, help="World size")
    args = parser.parse_args()
    main(args)
