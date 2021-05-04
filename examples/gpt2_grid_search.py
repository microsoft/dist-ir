import argparse
from collections import defaultdict, OrderedDict
import csv
import logging
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import Pool
from transformers import GPT2Tokenizer
import torch

import dist_ir
from dist_ir.importer import import_from_onnx
from dist_ir.ir import FunctionMaker, cpprint, pformat, Device, Topology, Value
from dist_ir.ir.type import Float32, Tensor
from dist_ir.executor import (
    CostModel,
    SequentialExecutor,
    PostTypeInferenceSimulator,
)
from dist_ir.transforms import gpt2_dhp_transform, filter_transform

NETWORK_BANDWIDTH_Gbps = 200
MODEL_PATH = "/lfs/1/keshav2/gpt2/model.onnx"


def add_devices_to_topology(topology, num_devices):
    for i in range(num_devices):
        topology.add_device("gpu")
    devices = topology.devices
    for i in range(0, len(devices)):
        for j in range(i + 1, len(devices)):
            topology.set_bandwidth(devices[i], devices[j], DGX_BANDWIDTH_GBPS)
    return topology


def to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x


def import_function_and_get_input_data(model_path, batch_size, default_device):
    function, input_data = import_from_onnx(
        model_path,
        name="GPT-2",
        default_device=default_device,
        parse_input_data=True,
    )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(
        "Here is some text to encode Hello World", add_special_tokens=True
    )
    input_ids = torch.tensor([[tokens] for _ in range(batch_size)])
    input_ids = to_numpy(input_ids)

    inputs_with_shapes = [
        Value(
            function.inputs[0].name,
            Tensor(
                dtype=Float32(),
                shape=tuple(input_ids.shape),
                device=default_device,
            ),
        )
    ]
    inputs_with_shapes += list(input_data.keys())
    input_data = [input_ids] + list(input_data.values())
    return function, input_data


def simulate(config):
    (
        batch_size,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
    ) = config

    world_size = dp_degree * hp_degree * pp_degree

    topology = Topology()
    d0 = topology.add_device("gpu")
    function, input_data = import_function_and_get_input_data(
        MODEL_PATH, batch_size=batch_size, default_device=d0
    )

    for i in range(1, world_size + 1):
        topology.add_device("gpu")
        for j in range(0, i):
            topology.set_bandwidth(
                topology.devices[i], topology.devices[j], NETWORK_BANDWIDTH_Gbps
            )

    function = gpt2_dhp_transform(
        function,
        dp_degree,
        hp_degree,
        pp_degree,
        topology.devices,
        num_microbatches,
    )

    # Manual adjustments for horizontal parallelism
    for i in range(len(input_data)):
        if input_data[i].shape == (1,) and input_data[i][0] == 2304:
            input_data[i] = np.array([input_data[i][0] // hp_degree])

    ex = SequentialExecutor("numpy")
    function = ex.infer_types(function, input_data)
    input_types = (v.type for v in function.inputs)
    function, typed_input_values = filter_transform(
        function, set(["Send", "MPIBroadcast", "MPIScatter"])
    )
    input_types = (v.type for v in typed_input_values)
    simulator = PostTypeInferenceSimulator(CostModel(topology))
    simulation = simulator.interpret(function, input_types)
    distributed_running_time = max(
        [simulation.timestamps[d] for d in simulation.timestamps]
    )
    throughput = batch_size / distributed_running_time
    return throughput


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


def grid_search():
    all_cluster_sizes = [1, 2, 4, 8]
    all_batch_sizes = [64, 128, 256, 512]
    configs = []
    for batch_size in all_batch_sizes:
        for i, cluster_size in enumerate(all_cluster_sizes):
            all_degrees = get_all_degrees(cluster_size)
            for (dp_degree, hp_degree, pp_degree) in all_degrees:
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
                        assert num_microbatches == 1
                    configs.append(
                        (
                            batch_size,
                            dp_degree,
                            hp_degree,
                            pp_degree,
                            num_microbatches,
                        )
                    )

    with Pool() as p:
        results = p.map(simulate, configs)

    with open("grid_search_results.csv", "w", newline="") as f:
        fieldnames = [
            "batch_size",
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
                dp_degree,
                hp_degree,
                pp_degree,
                num_microbatches,
            ) = config
            writer.writerow(
                {
                    "batch_size": batch_size,
                    "dp_degree": dp_degree,
                    "hp_degree": hp_degree,
                    "pp_degree": pp_degree,
                    "num_microbatches": num_microbatches,
                    "throughput": throughput,
                }
            )


if __name__ == "__main__":
    grid_search()
