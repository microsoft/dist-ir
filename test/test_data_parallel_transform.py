import numpy as np

from dist_ir.ir import Device, FunctionMaker
from dist_ir.ir.type import Float, Tensor
from dist_ir.transforms import DataParallelTransform
from dist_ir.executor import SequentialExecutor


def test_single_variable_partition():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    a = function.add_input_value("a", Tensor(Float(), (4, 4)))
    b = function.add_input_value("b", Tensor(Float(), (4, 4)))
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    function.finalize()
    transform = DataParallelTransform(
        batch_dims={"a": 0},
        reduction_params={"x": {"op_type": "Gather", "dim": 0, "device": d0}},
        devices=[d0, d1],
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

    # TODO is this really testing something useful?
    # assert transformed_function.is_op("Scatter/a")
    # assert transformed_function.is_op("Broadcast/b")
    # assert transformed_function.is_op("Pmap_#0")
    # assert transformed_function.is_op("Gather/x")


def test_double_variable_partition():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    a = function.add_input_value("a", Tensor(Float(), (4, 4)))
    b = function.add_input_value("b", Tensor(Float(), (4, 4)))
    c = function.add_input_value("c", Tensor(Float(), (4, 4)))
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    y = function.add_op("MatMul", "MatMul1", inputs=[x, c], output_names=["y"])
    function.finalize()
    transform = DataParallelTransform(
        batch_dims={"a": 0, "c": 0},
        reduction_params={"y": {"op_type": "Gather", "dim": 0, "device": d0}},
        devices=[d0, d1],
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

    # assert transformed_function.is_op("Scatter/a")
    # assert transformed_function.is_op("Broadcast/b")
    # assert transformed_function.is_op("Scatter/c")
    # assert transformed_function.is_op("Pmap_#0")
    # assert transformed_function.is_op("Gather/y")


def test_mnist():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    batch_size = 16
    x = function.add_input_value("x", Tensor(Float(), (batch_size, 4)))
    z = function.add_input_value("z", Tensor(Float(), (batch_size, 1)))
    wA = function.add_input_value("wA", Tensor(Float(), (4, 2)))
    wB = function.add_input_value("wB", Tensor(Float(), (2, 1)))
    a = function.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = function.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    l = function.add_op(
        "Loss", "Loss", inputs=[y, z], attributes={"N": batch_size}, output_names=["l"]
    )
    dl = function.add_op(
        "LossGrad",
        "LossGrad",
        inputs=[y, z],
        attributes={"N": batch_size},
        output_names=["dl"],
    )
    da, dwB = function.add_op(
        "MatMulGrad", "MatMul1Grad", inputs=[a, wB, dl], output_names=["da", "dwB"]
    )
    dx, dwA = function.add_op(
        "MatMulGrad", "MatMul0Grad", inputs=[x, wA, da], output_names=["dx", "dwA"]
    )
    function.set_outputs([l, dwA, dwB])
    function = function.finalize()
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

    ex = SequentialExecutor("numpy")
    _x = np.arange(batch_size * 4).reshape((batch_size, 4))
    _z = np.ones((batch_size, 1))
    _wA = np.ones((4, 2))
    _wB = np.ones((2, 1))
    orig_res = ex.compute(
        function,
        {x: _x, z: _z, wA: _wA, wB: _wB},
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

    (ls, dwAs, dwBs) = transformed_function.outputs
    assert np.array_equal(orig_res[l], np.concatenate(transformed_res[ls], axis=0))
    assert np.array_equal(orig_res[dwA], transformed_res[dwAs][0])
    assert np.array_equal(orig_res[dwB], transformed_res[dwBs][0])


if __name__ == "__main__":
    test_mnist()
