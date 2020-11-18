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
    b = module.add_input_value("b", Tensor(Float(), (4, 2)))
    x = module.add_op("Add", "Add0", inputs=[a, b], output_names=["x"])
    with pytest.raises(ValueError):
        infer_shapes(module)


def test_broadcast():
    module = Module()

    a = module.add_input_value("a", Tensor(Float(), (4, 4)))
    a0, a1 = module.add_op(
        "Broadcast",
        "Broadcast/a",
        inputs=[a],
        attributes={"devices": [0, 1]},
        output_names=["a0", "a1"],
    )
    infer_shapes(module)
    assert a0.type.shape == (4, 4)
    assert a1.type.shape == (4, 4)


def test_matmul_valid():
    module = Module()

    a = module.add_input_value("a", Tensor(Float(), (8, 4)))
    b = module.add_input_value("b", Tensor(Float(), (4, 2)))
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    infer_shapes(module)
    assert x.type.shape == (8, 2)


def test_matmul_invalid():
    module = Module()

    a = module.add_input_value("a", Tensor(Float(), (8, 8)))
    b = module.add_input_value("b", Tensor(Float(), (4, 2)))
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    with pytest.raises(ValueError):
        infer_shapes(module)


def test_matmul_grad():
    module = Module()

    x = module.add_input_value("x", Tensor(Float(), (8, 4)))
    w = module.add_input_value("w", Tensor(Float(), (4, 2)))
    l = module.add_input_value("l", Tensor(Float(), (8,)))
    dx, dw = module.add_op(
        "MatMulGrad", "MatMulGrad0", inputs=[x, w, l], output_names=["dx", "dw"]
    )
    infer_shapes(module)
    assert dx.type.shape == x.type.shape
    assert dw.type.shape == w.type.shape


def test_pmap():
    module = Module()

    submodule = Module()
    a = submodule.add_input_value("a", Tensor(Float(), (8, 4)))
    b = submodule.add_input_value("b", Tensor(Float(), (4, 2)))
    x = submodule.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])

    pmap_inputs = [
        module.add_input_value("a_0", Tensor(Float(), (4, 4))),
        module.add_input_value("a_1", Tensor(Float(), (4, 4))),
        module.add_input_value("b_0", Tensor(Float(), (4, 2))),
        module.add_input_value("b_1", Tensor(Float(), (4, 2))),
    ]
    pmap_output_names = ["x_0", "x_1"]
    value_name_map = {
        0: {
            "a": "a_0",
            "b": "b_0",
            "x": "x_0",
        },
        1: {
            "a": "a_1",
            "b": "b_1",
            "x": "x_1",
        },
    }

    (x_0, x_1) = module.add_op(
        "Pmap",
        inputs=pmap_inputs,
        attributes={"devices": [0, 1]},
        submodules=[submodule],
        metadata={"value_name_map": value_name_map},
        output_names=pmap_output_names,
    )

    infer_shapes(module)

    assert x_0.type.shape == (4, 2)
    assert x_1.type.shape == (4, 2)


def test_scatter():
    module = Module()

    a = module.add_input_value("a", Tensor(Float(), (4, 4)))
    a0, a1 = module.add_op(
        "Scatter",
        "Scatter/a",
        inputs=[a],
        attributes={"split_dim": 0, "num_splits": 2, "devices": [0, 1]},
        output_names=["a0", "a1"],
    )
    infer_shapes(module)
    assert a0.type.shape == (2, 4)
    assert a1.type.shape == (2, 4)
