from dist_ir.ir import Module
from dist_ir.ir.type import Float, Tensor
from dist_ir.ir.device import Device
from dist_ir.transforms import DataParallelTransform

# TODO test on actual inputs using sequential executor


def test_single_variable_partition():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    a = module.add_input_value("a", Tensor(Float(), (4, 4)))
    b = module.add_input_value("b", Tensor(Float(), (4, 4)))
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    transform = DataParallelTransform(partition_map={"a": 0}, devices=[d0, d1])
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
    assert transformed_module.is_op("Allreduce/x")


def test_double_variable_partition():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    a = module.add_input_value("a", Tensor(Float(), (4, 4)))
    b = module.add_input_value("b", Tensor(Float(), (4, 4)))
    c = module.add_input_value("c", Tensor(Float(), (4, 4)))
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    y = module.add_op("MatMul", "MatMul1", inputs=[x, c], output_names=["y"])
    transform = DataParallelTransform(partition_map={"a": 0, "c": 0}, devices=[d0, d1])
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
    assert transformed_module.is_op("Allreduce/y")


def test_mnist():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = module.add_input_value("x", Tensor(Float(), (16, 4)))
    z = module.add_input_value("z", Tensor(Float(), (16,)))
    wA = module.add_input_value("wA", Tensor(Float(), (4, 4)))
    wB = module.add_input_value("wB", Tensor(Float(), (4, 4)))
    a = module.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = module.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    l = module.add_op("Loss", "Loss", inputs=[y, z], output_names=["l"])
    dl = module.add_op("LossGrad", "LossGrad", inputs=[y, z], output_names=["dl"])
    da, dwB = module.add_op(
        "MatMulGrad", "MatMul1Grad", inputs=[a, wB, dl], output_names=["da", "dwB"]
    )
    dx, dwA = module.add_op(
        "MatMulGrad", "MatMul0Grad", inputs=[x, wA, da], output_names=["dx", "dwA"]
    )
    transform = DataParallelTransform(partition_map={"x": 0, "z": 0}, devices=[d0, d1])
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

    assert transformed_module.is_op("Scatter/x")
    assert transformed_module.is_op("Scatter/z")
    assert transformed_module.is_op("Broadcast/wA")
    assert transformed_module.is_op("Broadcast/wB")
    assert transformed_module.is_op("Pmap_#0")
    assert transformed_module.is_op("Allreduce/l")
    assert transformed_module.is_op("Allreduce/dx")
    assert transformed_module.is_op("Allreduce/dwA")
    assert transformed_module.is_op("Allreduce/dwB")


if __name__ == "__main__":
    test_mnist()
