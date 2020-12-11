import numpy as np

from dist_ir.ir import Device, Module
from dist_ir.ir.type import Float, Tensor
from dist_ir.transforms import PipelineParallelTransform
from dist_ir.executor import SequentialExecutor
import pipeline_parallel_utils as utils


def test_mnist_fw_bw():
    (module, partition_map) = utils.construct_module_and_partition_map()
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
        },
        partition_map=partition_map,
        schedule=schedule,
    )
    transformed_module = transform.apply(module)
    transformed_module.finalize()

    print("-" * 88)
    print("Original module")
    print("-" * 88)
    print(module)
    print()
    print("-" * 88)
    print("Transformed module")
    print("-" * 88)
    print(transformed_module)

    batch_size = 16
    ex = SequentialExecutor("numpy")
    _x = np.arange(batch_size * 4).reshape((batch_size, 4))
    _z = np.ones((batch_size, 1))
    _wA = np.ones((4, 2))
    _wB = np.ones((2, 1))
    orig_res = ex.compute(
        module,
        {"x": _x, "z": _z, "wA": _wA, "wB": _wB},
    )

    transformed_res = ex.compute(
        transformed_module,
        {"x": _x, "z": _z, "wA": _wA, "wB": _wB},
    )

    print("-" * 88)
    print("Original module results")
    print("-" * 88)
    for k, v in orig_res.items():
        print(k)
        print(v)
        print()
    print()
    print("-" * 88)
    print("Transformed module results")
    print("-" * 88)
    for k, v in transformed_res.items():
        print(k)
        print(v)
        print()

    assert np.array_equal(orig_res["l"], transformed_res["l"])
    assert np.array_equal(orig_res["dwA"], transformed_res["dwA"])
    assert np.array_equal(orig_res["dwB"], transformed_res["dwB"])


if __name__ == "__main__":
    test_mnist_fw_bw()
