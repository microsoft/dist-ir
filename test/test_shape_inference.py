import pytest

from dist_ir.ir import Module
from dist_ir.executor.shape_inference import infer_shapes
from dist_ir.ir.type import Float, Tensor


def test_add_valid():
    module = Module()

    a = module.add_input_value("a", Tensor(Float(), (4, 4)))
    b = module.add_input_value("b", Tensor(Float(), (4, 4)))
    x = module.add_op("Add", "Add0", inputs=[a, b], output_names=["x"])
    infer_shapes(module)
    assert x.type.shape == (4, 4)


def test_add_invalid():
    module = Module()

    a = module.add_input_value("a", Tensor(Float(), (8, 4)))
    b = module.add_input_value("b", Tensor(Float(), (4, 4)))
    x = module.add_op("Add", "Add0", inputs=[a, b], output_names=["x"])
    with pytest.raises(ValueError):
        infer_shapes(module)


def test_matmul_valid():
    module = Module()

    a = module.add_input_value("a", Tensor(Float(), (8, 4)))
    b = module.add_input_value("b", Tensor(Float(), (4, 8)))
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    infer_shapes(module)
    assert x.type.shape == (8, 8)


def test_matmul_invalid():
    module = Module()

    a = module.add_input_value("a", Tensor(Float(), (8, 8)))
    b = module.add_input_value("b", Tensor(Float(), (4, 8)))
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    with pytest.raises(ValueError):
        infer_shapes(module)
