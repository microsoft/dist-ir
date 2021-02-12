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
from dist_ir.transforms import (
    parallel_transform_3d,
    hybrid_transform_unrolled,
    PipeDreamScheduler,
)


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
    hidden_dim = args.hidden_dim
    for i in range(args.num_hidden_layers):
        w = function.add_input_value(
            f"w{chr(ord('A')+i)}",
            Tensor(dtype=Float(), shape=(input_dim, hidden_dim), device=devices[0]),
        )
        input_dim = hidden_dim
        weights.append(w)
    w = function.add_input_value(
        f"w{chr(ord('A')+i+1)}",
        Tensor(dtype=Float(), shape=(hidden_dim, args.output_dim), device=devices[0]),
    )
    weights.append(w)

    a = x
    for i, weight in enumerate(weights):
        y = function.add_op("MatMul", inputs=[a, weight], output_names=[f"y{i}"])
        a = function.add_op("Relu", inputs=[y], output_names=[f"a{i}"])
        # TODO: Add activation function?

    l = function.add_op(
        "Loss", inputs=[a, z], attributes={"N": args.batch_size}, output_names=["l"]
    )
    dl = function.add_op(
        "LossGrad",
        inputs=[a, z],
        attributes={"N": args.batch_size},
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


def partition(args, function, device_tree):
    num_blocks = args.num_hidden_layers + 1
    assert num_blocks % args.pp_degree == 0
    num_blocks_per_device = num_blocks // args.pp_degree
    partition_maps = {}
    root_device = list(device_tree.keys())[0]
    for dp_root_device in device_tree[root_device]:
        partition_maps[dp_root_device] = {}
        for hp_root_device in device_tree[root_device][dp_root_device]:
            partition_map = {}
            for i, device in enumerate(device_tree[root_device][dp_root_device]):
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
            partition_maps[dp_root_device][hp_root_device] = partition_map
    return partition_maps


def get_device_tree(args, devices):
    dp_size = args.world_size // args.dp_degree
    hp_size = dp_size // args.hp_degree
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
                for j in range(args.hp_degree)
            }
            for i in range(args.dp_degree)
        }
    }
    return device_tree


def main(args):
    if not args.num_hidden_layers % 2 == 1:
        raise ValueError(
            "Must have odd number of hidden layers to support horizontal parallelism"
        )
    if not args.dp_degree * args.pp_degree * args.hp_degree == args.world_size:
        raise ValueError(
            "Product of data parallel, pipeline parallel, and horizontal parallel "
            "degrees must be the world size"
        )
    if not args.num_hidden_layers + 1 >= args.pp_degree:  # args.world_size:
        raise ValueError(
            "Must have at least as many layers as pipeline parallel devices"
        )

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

    device_tree = get_device_tree(args, devices)
    transformed_function = parallel_transform_3d(
        args, function, device_tree, args.num_microbatches
    )
    transformed_function = infer_types(
        transformed_function, transformed_function.inputs
    )
    cpprint(transformed_function, width=5000)
    transformed_res = ex.compute(transformed_function, input_data)
    for i, a in enumerate(function.outputs):
        for j, b in enumerate(transformed_function.outputs):
            if a.name == b.name and a.type.shape == b.type.shape:
                try:
                    np.testing.assert_array_almost_equal(res[i], transformed_res[j])
                except AssertionError as e:
                    print(f"Outputs {a} and {b} do not match!")
                    print(res[i])
                    print()
                    print(transformed_res[j])
                    print()
                    print("-" * 100)
                    print()
                break
    # for a, b in zip(res, transformed_res):
    #    np.testing.assert_array_almost_equal(a, b)
    """
    simulator = Simulator(CostModel(topology, device_speeds))
    simulation = simulator.interpret(
        transformed_function, (v.type for v in transformed_function.inputs)
    )
    """


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
        "--num_hidden_layers",
        type=int,
        help="Number of hidden layers",
    )
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--output_dim", type=int, default=64, help="Output dimension")
    parser.add_argument("--world_size", type=int, default=8, help="World size")
    parser.add_argument("--dp_degree", type=int, default=2, help="Data parallel degree")
    parser.add_argument(
        "--pp_degree", type=int, default=2, help="Pipeline parallel degree"
    )
    parser.add_argument(
        "--hp_degree", type=int, default=2, help="Horizontal parallel degree"
    )
    args = parser.parse_args()
    main(args)
