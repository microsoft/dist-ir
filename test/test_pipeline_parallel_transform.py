import numpy as np

from dist_ir.ir import Device, Function
from dist_ir.ir.type import Float, Tensor
from dist_ir.transforms import PipelineParallelTransform
from dist_ir.executor import SequentialExecutor
import pipeline_parallel_utils as utils


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
        batch_dims={"x": 0, "z": 0},
        reduction_params={
            "dwB": {"op_type": "Add"},
            "dwA": {"op_type": "Add"},
            "l": {"op_type": "Concat", "dim": 0},
            "dx": {"op_type": "Concat", "dim": 0},
        },
        partition_map=partition_map,
        schedule=schedule,
    )
    transformed_function = transform.apply(function)

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    print(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    print(transformed_function)

    batch_size = 16
    ex = SequentialExecutor("numpy")
    _x = np.arange(batch_size * 4).reshape((batch_size, 4))
    _z = np.ones((batch_size, 1))
    _wA = np.ones((4, 2))
    _wB = np.ones((2, 1))
    orig_res = ex.compute(
        function,
        {
            function.inputs[0]: _x,
            function.inputs[1]: _z,
            function.inputs[2]: _wA,
            function.inputs[3]: _wB,
        },
    )

    transformed_res = ex.compute(
        transformed_function,
        {
            transformed_function.inputs[0]: _x,
            transformed_function.inputs[1]: _z,
            transformed_function.inputs[2]: _wA,
            transformed_function.inputs[3]: _wB,
        },
    )

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    for k, v in orig_res.items():
        print(k)
        print(v)
        print()
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    for k, v in transformed_res.items():
        print(k)
        print(v)
        print()

    for i in range(len(function.outputs)):
        np.testing.assert_array_almost_equal(
            orig_res[function.outputs[i]],
            transformed_res[transformed_function.outputs[i]],
        )


if __name__ == "__main__":
    test_mnist_fw_bw()
