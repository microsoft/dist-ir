import numpy as np

from dist_ir.ir import Module
from dist_ir.ir.type import Float, Tensor
from dist_ir.ir.device import Device
from dist_ir.transforms import DataParallelTransform
from dist_ir.executor import SequentialExecutor

# TODO test on actual inputs using sequential executor


def test_single_variable_partition():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    a = module.add_input_value("a", Tensor(Float(), (4, 4)))
    b = module.add_input_value("b", Tensor(Float(), (4, 4)))
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    module.finalize()
    transform = DataParallelTransform(
        batch_dims={"a": 0},
        reduction_params={"x": {"op_type": "Gather", "dim": 0, "device": d0}},
        devices=[d0, d1],
    )
    transformed_module = transform.apply(module)

    print("-" * 88)
    print("Original module")
    print("-" * 88)
    print(module)
    print()
    print("-" * 88)
    print("Transformed module")
    print("-" * 88)
    print(transformed_module)

    assert transformed_module.is_op("Scatter/a")
    assert transformed_module.is_op("Broadcast/b")
    assert transformed_module.is_op("Pmap_#0")
    assert transformed_module.is_op("Gather/x")


def test_double_variable_partition():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    a = module.add_input_value("a", Tensor(Float(), (4, 4)))
    b = module.add_input_value("b", Tensor(Float(), (4, 4)))
    c = module.add_input_value("c", Tensor(Float(), (4, 4)))
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    y = module.add_op("MatMul", "MatMul1", inputs=[x, c], output_names=["y"])
    module.finalize()
    transform = DataParallelTransform(
        batch_dims={"a": 0, "c": 0},
        reduction_params={"y": {"op_type": "Gather", "dim": 0, "device": d0}},
        devices=[d0, d1],
    )
    transformed_module = transform.apply(module)

    print("-" * 88)
    print("Original module")
    print("-" * 88)
    print(module)
    print()
    print("-" * 88)
    print("Transformed module")
    print("-" * 88)
    print(transformed_module)

    assert transformed_module.is_op("Scatter/a")
    assert transformed_module.is_op("Broadcast/b")
    assert transformed_module.is_op("Scatter/c")
    assert transformed_module.is_op("Pmap_#0")
    assert transformed_module.is_op("Gather/y")


def test_mnist():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    batch_size = 16
    x = module.add_input_value("x", Tensor(Float(), (batch_size, 4)))
    z = module.add_input_value("z", Tensor(Float(), (batch_size, 1)))
    wA = module.add_input_value("wA", Tensor(Float(), (4, 2)))
    wB = module.add_input_value("wB", Tensor(Float(), (2, 1)))
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
    dx, dwA = module.add_op(
        "MatMulGrad", "MatMul0Grad", inputs=[x, wA, da], output_names=["dx", "dwA"]
    )
    module.set_outputs([l, dwA, dwB])
    module.finalize()
    transform = DataParallelTransform(
        batch_dims={"x": 0, "z": 0},
        reduction_params={
            "l": {"op_type": "Gather", "dim": 0, "device": d0},
            "dx": {"op_type": "Gather", "dim": 0, "device": d0},
            "dwA": {"op_type": "Allreduce"},
            "dwB": {"op_type": "Allreduce"},
        },
        devices=[d0, d1],
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

    assert np.array_equal(orig_res["l"], np.concatenate(transformed_res["ls"], axis=0))
    assert np.array_equal(orig_res["dwA"], transformed_res["dwAs"][0])
    assert np.array_equal(orig_res["dwB"], transformed_res["dwBs"][0])

    """
    assert transformed_module.is_op("Scatter/x")
    assert transformed_module.is_op("Scatter/z")
    assert transformed_module.is_op("Broadcast/wA")
    assert transformed_module.is_op("Broadcast/wB")
    assert transformed_module.is_op("Pmap_#0")
    assert transformed_module.is_op("Gather/l")
    assert transformed_module.is_op("Gather/dx")
    assert transformed_module.is_op("Allreduce/dwA")
    assert transformed_module.is_op("Allreduce/dwB")
    """


if __name__ == "__main__":
    test_mnist()
