import numpy as np

from dist_ir.ir import Device, Module
from dist_ir.ir.type import Float, Tensor
from dist_ir.transforms import HorizontalParallelTransform
from dist_ir.executor import SequentialExecutor


def test_single_variable_partition():
    batch_size = 16
    input_dim = 32
    hidden_dim = 16
    output_dim = 8

    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = module.add_input_value("x", Tensor(Float(), (batch_size, input_dim)))
    wA = module.add_input_value("wA", Tensor(Float(), (input_dim, hidden_dim)))
    wB = module.add_input_value("wB", Tensor(Float(), (hidden_dim, output_dim)))
    a = module.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = module.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    module.finalize()
    transform = HorizontalParallelTransform(
        op_names=("MatMul0",),
        param_dims={"wA": 1},
        reduction_params={"a": {"op_type": "Gather", "dim": 1, "device": d0}},
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

    ex = SequentialExecutor("numpy")
    _x = np.random.normal(size=(batch_size, input_dim))
    _wA = np.random.normal(size=(input_dim, hidden_dim))
    _wB = np.random.normal(size=(hidden_dim, output_dim))
    orig_res = ex.compute(module, {"x": _x, "wA": _wA, "wB": _wB})

    transformed_res = ex.compute(transformed_module, {"x": _x, "wA": _wA, "wB": _wB})

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

    assert np.array_equal(orig_res["y"], transformed_res["y"])


def test_double_variable_counter_example():
    batch_size = 16
    input_dim = 32
    hidden_dim = 16
    output_dim = 8

    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = module.add_input_value("x", Tensor(Float(), (batch_size, input_dim)))
    wA = module.add_input_value("wA", Tensor(Float(), (input_dim, hidden_dim)))
    wB = module.add_input_value("wB", Tensor(Float(), (hidden_dim, output_dim)))
    a = module.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = module.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    module.finalize()
    transform = HorizontalParallelTransform(
        op_names=("MatMul0", "MatMul1"),
        param_dims={"wA": 1, "wB": 0},
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

    ex = SequentialExecutor("numpy")
    _x = np.random.normal(size=(batch_size, input_dim))
    _wA = np.random.normal(size=(input_dim, hidden_dim))
    _wB = np.random.normal(size=(hidden_dim, output_dim))
    orig_res = ex.compute(module, {"x": _x, "wA": _wA, "wB": _wB})

    transformed_res = ex.compute(transformed_module, {"x": _x, "wA": _wA, "wB": _wB})

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

    assert not np.array_equal(orig_res["y"], transformed_res["y"])


def test_data_parallel():
    batch_size = 16
    input_dim = 32
    hidden_dim = 16
    output_dim = 8

    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = module.add_input_value("x", Tensor(Float(), (batch_size, input_dim)))
    wA = module.add_input_value("wA", Tensor(Float(), (input_dim, hidden_dim)))
    wB = module.add_input_value("wB", Tensor(Float(), (hidden_dim, output_dim)))
    a = module.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = module.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    module.finalize()
    transform = HorizontalParallelTransform(
        op_names=("MatMul0", "MatMul1"),
        param_dims={"x": 0},
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

    ex = SequentialExecutor("numpy")
    _x = np.random.normal(size=(batch_size, input_dim))
    _wA = np.random.normal(size=(input_dim, hidden_dim))
    _wB = np.random.normal(size=(hidden_dim, output_dim))
    orig_res = ex.compute(module, {"x": _x, "wA": _wA, "wB": _wB})

    transformed_res = ex.compute(transformed_module, {"x": _x, "wA": _wA, "wB": _wB})

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

    assert np.array_equal(orig_res["y"], transformed_res["y"])


if __name__ == "__main__":
    test_data_parallel()
