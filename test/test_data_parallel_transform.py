import pytest

from dist_ir.ir import Module
from dist_ir.transforms import DataParallelTransform
from dist_ir.ir.type import Float, Tensor


def test_single_matmul():
    module = Module()

    a = module.add_input_value("a", Tensor(Float(), (4, 4)))
    b = module.add_input_value("b", Tensor(Float(), (4, 4)))
    x = module.add_op("MatMul", "MatMul", inputs=[a, b], output_names=["x"])
    transform = DataParallelTransform(
        partitioned_input_name="a", partition_dim=0, num_partitions=2
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


def test_double_matmul():
    module = Module()

    a = module.add_input_value("a", Tensor(Float(), (4, 4)))
    b = module.add_input_value("b", Tensor(Float(), (4, 4)))
    c = module.add_input_value("c", Tensor(Float(), (4, 4)))
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    y = module.add_op("MatMul", "MatMul1", inputs=[x, c], output_names=["y"])
    transform = DataParallelTransform(
        partitioned_input_name="a", partition_dim=0, num_partitions=2
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


if __name__ == "__main__":
    test_double_matmul()
