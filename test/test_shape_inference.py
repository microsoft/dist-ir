import pytest

from dist_ir.ir import Module
from dist_ir.transforms import DataParallelTransform
from dist_ir.executor.shape_inference import infer_shapes
from dist_ir.ir.type import Float, Tensor


def test_matmul():
    module = Module()

    a = module.add_input_value("a", Tensor(Float(), (4, 4)))
    b = module.add_input_value("b", Tensor(Float(), (4, 4)))
    x = module.add_op("MatMul", "MatMul", inputs=[a, b], output_names=["x"])

    print("-" * 88)
    print("Original module")
    print("-" * 88)
    print(module)
    print()

    infer_shapes(module)

    print("-" * 88)
    print("Module after shape inference")
    print("-" * 88)
    print(module)


def test_backward_pass():
    module = Module()

    x = module.add_input_value("x", Tensor(Float(), (16, 4)))
    z = module.add_input_value("z", Tensor(Float(), (16)))
    wA = module.add_input_value("wA", Tensor(Float(), (4, 4)))
    wB = module.add_input_value("wB", Tensor(Float(), (4, 4)))
    a = module.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = module.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    l = module.add_op("Loss", "Loss", inputs=[y, z], output_names=["l"])
    dl = module.add_op("LossGrad", "LossGrad", inputs=[y, z], output_names=["dl"])
    da, dwB = module.add_op(
        "MatMulGrad", "MatMul1Grad", inputs=[a, wB, dl], output_names=["da", "dWB"]
    )
    dx, dwA = module.add_op(
        "MatMulGrad", "MatMul0Grad", inputs=[x, wA, da], output_names=["dx", "dwA"]
    )
    transform = DataParallelTransform(
        partitioned_input_name="x", partition_dim=0, num_partitions=2
    )
    transformed_module = transform.apply(module)

    print("-" * 88)
    print("Original module")
    print("-" * 88)
    print(transformed_module)
    print()

    infer_shapes(transformed_module)
    print("-" * 88)
    print("Module after shape inference")
    print("-" * 88)
    print(transformed_module)


if __name__ == "__main__":
    test_backward_pass()
