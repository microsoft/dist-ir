import pytest

from dist_ir.ir import Module
from dist_ir.ir.type import Float, Tensor
from dist_ir.ir.device import Device
from dist_ir.transforms import PipelineParallelTransform
from dist_ir.executor.shape_inference import infer_shapes


def test_mnist():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = module.add_input_value("x", Tensor(dtype=Float(), shape=(16, 4), device=d0))
    z = module.add_input_value("z", Tensor(dtype=Float(), shape=(16,), device=d0))
    wA = module.add_input_value("wA", Tensor(dtype=Float(), shape=(4, 4), device=d0))
    wB = module.add_input_value("wB", Tensor(dtype=Float(), shape=(4, 4), device=d0))
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

    infer_shapes(module)
    transform = PipelineParallelTransform(
        num_microbatches=2, inputs_to_partition={"x": 0, "z": 0}, schedule=schedule
    )
    transformed_module = transform.apply(module)
    infer_shapes(transformed_module)

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
    test_mnist()
