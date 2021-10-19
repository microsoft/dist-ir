# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from dist_ir.ir import cpprint
from dist_ir.transforms import PipelineParallelTransform
from dist_ir.executor import ConcreteValue, sequentially_execute
from . import pipeline_parallel_utils as utils


def test_mnist_fw_bw():
    (function, partition_map) = utils.construct_function_and_partition_map()
    (d0, d1) = sorted(set(partition_map.values()))
    stages = list(partition_map.keys())
    schedule = [
        {d0: (stages[0], 0)},
        {d0: (stages[0], 1), d1: (stages[1], 0)},
        {d1: (stages[2], 0)},
        {d0: (stages[3], 0), d1: (stages[1], 1)},
        {d1: (stages[2], 1)},
        {d0: (stages[3], 1)},
    ]
    transform = PipelineParallelTransform(
        num_microbatches=2,
        batch_dims={function.inputs[0]: 0, function.inputs[1]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "Concat", "dim": 0},  # l
            function.outputs[1]: {"op_type": "Add"},  # dwB
            function.outputs[2]: {"op_type": "Concat", "dim": 0},  # dx
            function.outputs[3]: {"op_type": "Add"},  # dwA
        },
        partition_map=partition_map,
        schedule=schedule,
    )
    transformed_function = transform.apply(function)

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    cpprint(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    cpprint(transformed_function)

    batch_size = 16
    _x = np.arange(batch_size * 4).reshape((batch_size, 4))
    _z = np.ones((batch_size, 1))
    _n = (batch_size,)
    _wA = np.ones((4, 2))
    _wB = np.ones((2, 1))
    # TODO output devices are correct
    inputs = [ConcreteValue(v, None) for v in [_x, _z, _n, _wA, _wB]]
    orig_res = sequentially_execute(function, inputs)
    transformed_res = sequentially_execute(transformed_function, inputs)

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    print(orig_res)
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    print(transformed_res)
    print()

    for a, b in zip(orig_res, transformed_res):
        np.testing.assert_array_almost_equal(a.val, b.val)


if __name__ == "__main__":
    test_mnist_fw_bw()
