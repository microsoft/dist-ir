import argparse
from collections import defaultdict, OrderedDict
import csv
import logging
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import Pool

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


def get_all_degrees(n):
    all_degrees = []
    d = 1
    h = 1
    p = 1
    while d <= n:
        h = 1
        p = 1
        if d * h * p == n:
            all_degrees.append((d, h, p))
            break
        while h <= n:
            p = 1
            if d * h * p == n:
                all_degrees.append((d, h, p))
                break
            while p <= n:
                if d * h * p == n:
                    all_degrees.append((d, h, p))
                    break
                p *= 2
            h *= 2
        d *= 2
    return all_degrees


def run_experiment(config):
    (
        batch_size,
        input_dim,
        num_hidden_layers,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
    ) = config
    hidden_dim = input_dim
    output_dim = hidden_dim
    topology = Topology()
    d0 = topology.add_device("gpu")
    function = mlp(batch_size, input_dim, hidden_dim, output_dim, num_hidden_layers, d0)
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
    """
    transformed_function, typed_input_values = steady_state_transform(
        transformed_function
    )
    transformed_function = infer_types(
        transformed_function, typed_input_values
    )
    """
    simulator = Simulator(CostModel(topology))
    simulation = simulator.interpret(
        transformed_function,
        (v.type for v in transformed_function.inputs),
    )
    distributed_running_time = max(
        [simulation.timestamps[d] for d in simulation.timestamps]
    )
    # communication_overhead = measure_communication_overhead(simulation)
    throughput = batch_size / distributed_running_time
    return throughput


def grid_search():
    input_dim = 8192
    hidden_dim = input_dim
    output_dim = input_dim
    all_cluster_sizes = [1, 2, 4, 8, 16, 32]  # [64, 128, 512, 1024]
    all_num_hidden_layers = [64]  # [4, 8, 16, 32]
    all_batch_sizes = [8192]  # [512, 1024, 2048, 4096, 8192]
    configs = []
    for num_hidden_layers in all_num_hidden_layers:
        for batch_size in all_batch_sizes:
            for i, cluster_size in enumerate(all_cluster_sizes):
                all_degrees = get_all_degrees(cluster_size)
                for (dp_degree, hp_degree, pp_degree) in all_degrees:
                    if num_hidden_layers % pp_degree != 0:
                        continue
                    dp_batch_size = batch_size // dp_degree
                    if pp_degree == 1:
                        all_num_microbatches = [1]
                    else:
                        all_num_microbatches = [
                            int(2 ** k)
                            for k in range(1, int(np.floor(np.log2(dp_batch_size) / 2)))
                        ]
                    for num_microbatches in all_num_microbatches:
                        if pp_degree == 1:
                            num_microbatches == 1
                        configs.append(
                            (
                                batch_size,
                                input_dim,
                                num_hidden_layers,
                                dp_degree,
                                hp_degree,
                                pp_degree,
                                num_microbatches,
                            )
                        )

    with Pool() as p:
        results = p.map(run_experiment, configs)
        # results = map(run_experiment, configs)

    with open("grid_search_results.csv", "w", newline="") as f:
        fieldnames = [
            "dp_degree",
            "hp_degree",
            "pp_degree",
            "num_microbatches",
            "throughput",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for config, throughput in zip(configs, results):
            (
                batch_size,
                input_dim,
                num_hidden_layers,
                dp_degree,
                hp_degree,
                pp_degree,
                num_microbatches,
            ) = config
            writer.writerow(
                {
                    "dp_degree": dp_degree,
                    "hp_degree": hp_degree,
                    "pp_degree": pp_degree,
                    "num_microbatches": num_microbatches,
                    "throughput": throughput,
                }
            )


if __name__ == "__main__":
    grid_search()
