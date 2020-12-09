import numpy as np

from dist_ir.ir import Device, Module
from dist_ir.ir.type import Float, Tensor
from dist_ir.transforms import PipelineParallelTransform
from dist_ir.executor import SequentialExecutor


def test_mnist_fw():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    batch_size = 16
    x = module.add_input_value(
        "x", Tensor(dtype=Float(), shape=(batch_size, 4), device=d0)
    )
    z = module.add_input_value(
        "z", Tensor(dtype=Float(), shape=(batch_size, 1), device=d0)
    )
    wA = module.add_input_value("wA", Tensor(dtype=Float(), shape=(4, 2), device=d0))
    wB = module.add_input_value("wB", Tensor(dtype=Float(), shape=(2, 1), device=d0))
    a = module.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = module.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    l = module.add_op(
        "Loss", "Loss", inputs=[y, z], attributes={"N": batch_size}, output_names=["l"]
    )
    module.set_outputs([l])
    module.finalize()

    partition_map = {
        "MatMul0": d0,
        "MatMul1": d1,
        "Loss": d1,
    }

    schedule = [
        {d0: ("MatMul0", 0)},
        {d0: ("MatMul0", 1), d1: ("MatMul1", 0)},
        {d0: ("Loss", 0), d1: ("MatMul1", 1)},
        {d1: ("Loss", 1)},
    ]

    transform = PipelineParallelTransform(
        num_microbatches=2,
        batch_dims={"x": 0, "z": 0},
        reduction_params={
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

    ex = SequentialExecutor("numpy")
    _x = np.arange(16 * 4).reshape((16, 4))
    _z = np.ones((16, 1))
    _wA = np.ones((4, 2))
    _wB = np.ones((2, 1))
    orig_res = ex.compute(
        module,
        {"x": _x, "z": _z, "wA": _wA, "wB": _wB},
    )

    # TODO: Assert output matches between original module and transformed module
    transformed_res = ex.compute(
        transformed_module,
        {"x": _x, "z": _z, "wA": _wA, "wB": _wB},
    )

    print("-" * 88)
    print("Original module results")
    print("-" * 88)
    print(orig_res)
    print()
    print("-" * 88)
    print("Transformed module results")
    print("-" * 88)
    print(transformed_res)

    assert np.array_equal(orig_res["l"], transformed_res["l"])


def test_mnist_fw_bw():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    batch_size = 16
    x = module.add_input_value(
        "x", Tensor(dtype=Float(), shape=(batch_size, 4), device=d0)
    )
    z = module.add_input_value(
        "z", Tensor(dtype=Float(), shape=(batch_size, 1), device=d0)
    )
    wA = module.add_input_value("wA", Tensor(dtype=Float(), shape=(4, 2), device=d0))
    wB = module.add_input_value("wB", Tensor(dtype=Float(), shape=(2, 1), device=d0))
    a = module.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = module.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    l = module.add_op(
        "Loss", "Loss", inputs=[y, z], attributes={"N": batch_size}, output_names=["l"]
    )
    dl = module.add_op(
        "LossGrad",
        "LossGrad",
        inputs=[y, z],
        attributes={"N": batch_size},
        output_names=["dl"],
    )
    da, dwB = module.add_op(
        "MatMulGrad", "MatMul1Grad", inputs=[a, wB, dl], output_names=["da", "dwB"]
    )
    _, dwA = module.add_op(
        "MatMulGrad", "MatMul0Grad", inputs=[x, wA, da], output_names=["dx", "dwA"]
    )
    module.set_outputs([l, dwA, dwB])
    module.finalize()

    partition_map = {
        "MatMul0": d0,
        "MatMul1": d1,
        "Loss": d1,
        "LossGrad": d1,
        "MatMul1Grad": d1,
        "MatMul0Grad": d0,
    }

    schedule = [
        {d0: ("MatMul0", 0)},
        {d0: ("MatMul0", 1), d1: ("MatMul1", 0)},
        {d1: ("Loss", 0)},
        {d1: ("LossGrad", 0)},
        {d1: ("MatMul1Grad", 0)},
        {d0: ("MatMul0Grad", 0), d1: ("MatMul1", 1)},
        {d1: ("Loss", 1)},
        {d1: ("LossGrad", 1)},
        {d1: ("MatMul1Grad", 1)},
        {d0: ("MatMul0Grad", 1)},
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
