import argparse
from collections import defaultdict, OrderedDict
import logging
import numpy as np
import time

import dist_ir
from dist_ir.importer import import_from_onnx, parse_tensor_from_file
from dist_ir.ir import FunctionMaker, cpprint, pformat, Device, Topology, Value
from dist_ir.executor import infer_types, SequentialExecutor, Simulator
from dist_ir.executor.cost_model import CostModel
from dist_ir.ir.type import Bool, Float, Int64, Tensor
from dist_ir.transforms import (
    parallel_transform_3d,
    steady_state_transform,
    PipeDreamScheduler,
)

DGX_BANDWIDTH_GBPS = 200


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
    for i in range(args.num_hidden_layers - 1):
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


def main(args):
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if not args.num_hidden_layers % 2 == 0:
        raise ValueError(
            "Must have even number of hidden layers to support horizontal parallelism"
        )
    if not args.num_hidden_layers >= args.pp_degree:
        raise ValueError(
            "Must have at least as many layers as pipeline parallel devices"
        )

    topology = Topology()
    world_size = args.dp_degree * args.hp_degree * args.pp_degree
    devices = [topology.add_device("gpu") for i in range(world_size + 1)]
    for i in range(0, len(devices)):
        for j in range(i + 1, len(devices)):
            topology.set_bandwidth(devices[i], devices[j], DGX_BANDWIDTH_GBPS)

    function = mlp(args, topology.devices)
    if args.verbose:
        cpprint(function)
    function = infer_types(function, function.inputs)
    print(f"Function has {len(function.ops)} ops at start")
    if args.verbose:
        cpprint(function)
    if args.evaluate:
        ex = SequentialExecutor("numpy")
        input_data = [
            np.random.normal(size=(inp.type.shape)) for inp in function.inputs
        ]
        res = ex.compute(function, input_data)

    start_time = time.time()
    transformed_function = parallel_transform_3d(
        function,
        args.dp_degree,
        args.hp_degree,
        args.pp_degree,
        devices,
        args.num_microbatches,
    )
    duration = time.time() - start_time
    print(f"3D transform time: {duration}")
    transformed_function = infer_types(
        transformed_function, transformed_function.inputs
    )
    print(f"Function has {len(transformed_function.ops)} ops after 3D transform")
    if args.verbose:
        cpprint(transformed_function, width=1000)
    if args.evaluate:
        transformed_res = ex.compute(transformed_function, input_data)
        for i, a in enumerate(function.outputs):
            for j, b in enumerate(transformed_function.outputs):
                if a.name == b.name and a.type.shape == b.type.shape:
                    try:
                        np.testing.assert_array_almost_equal(
                            res[i], transformed_res[j], decimal=2
                        )
                    except AssertionError as e:
                        print(f"Outputs {a} and {b} do not match!")
                        print(res[i])
                        print()
                        print(transformed_res[j])
                        print()
                        print(
                            f"Difference: {np.linalg.norm(res[i] - transformed_res[j])}"
                        )
                        print("-" * 100)
                        print()
                    break
    start_time = time.time()
    transformed_function, typed_input_values = steady_state_transform(
        transformed_function
    )
    duration = time.time() - start_time
    print(f"Steady state transform time: {duration}")
    transformed_function = infer_types(transformed_function, typed_input_values)
    if args.verbose:
        cpprint(transformed_function, width=1000)
    start_time = time.time()
    simulator = Simulator(CostModel(topology))
    print(f"Function has {len(transformed_function.ops)} ops after steady state")
    op_counts = defaultdict(lambda: 0)
    for op in transformed_function.ops:
        op_counts[op.op_type] += 1
    for op_type, count in op_counts.items():
        print(f"{op_type}: {count}")
    simulation = simulator.interpret(
        transformed_function, (v.type for v in transformed_function.inputs)
    )
    duration = time.time() - start_time
    print(f"Simulation time: {duration}")
    simulation.dump_chrome_trace("distributed_trace.json")


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
    parser.add_argument("--dp_degree", type=int, default=2, help="Data parallel degree")
    parser.add_argument(
        "--pp_degree", type=int, default=2, help="Pipeline parallel degree"
    )
    parser.add_argument(
        "--hp_degree", type=int, default=2, help="Horizontal parallel degree"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="If set, evaluate the function",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Verbose mode"
    )
    args = parser.parse_args()
    main(args)