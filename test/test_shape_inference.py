import pytest

from dist_ir.ir import Module
from dist_ir.ir.device import Device
from dist_ir.executor.shape_inference import infer_shapes
from dist_ir.ir.type import Float, Tensor, ValueTuple


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


def test_allreduce():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    xis = module.add_input_value(
        "xis",
        ValueTuple(
            (Tensor(Float(), (4, 4), device=d0), Tensor(Float(), (4, 4), device=d1))
        ),
    )
    xs = module.add_op(
        "Allreduce",
        "Allreduces/xis",
        inputs=[xis],
        output_names=["xs"],
    )
    infer_shapes(module)

    assert isinstance(xs.type, ValueTuple)
    for i, value_type in enumerate(xis.type.types):
        assert value_type.shape == xs.type.types[i].shape
        assert value_type.device == xs.type.types[i].device


def test_broadcast():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = module.add_input_value("x", Tensor(Float(), (4, 4)))
    xs = module.add_op(
        "Broadcast",
        "Broadcast/x",
        inputs=[x],
        attributes={"devices": [d0, d1]},
        output_names=["xs"],
    )
    infer_shapes(module)

    assert isinstance(xs.type, ValueTuple)
    assert xs.type.types[0].shape == (4, 4)
    assert xs.type.types[0].device == d0
    assert xs.type.types[1].shape == (4, 4)
    assert xs.type.types[1].device == d1


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

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    xs = module.add_input_value(
        "xs",
        ValueTuple(
            (Tensor(Float(), (8, 4), device=d0), Tensor(Float(), (8, 4), device=d1))
        ),
    )
    wAs = module.add_input_value(
        "wAs",
        ValueTuple(
            (Tensor(Float(), (4, 2), device=d0), Tensor(Float(), (4, 2), device=d1))
        ),
    )
    wBs = module.add_input_value(
        "wBs",
        ValueTuple(
            (Tensor(Float(), (2, 1), device=d0), Tensor(Float(), (2, 1), device=d1))
        ),
    )

    submodule = Module()
    x = submodule.add_input_value("x", Tensor(Float(), (16, 4)))
    wA = submodule.add_input_value("wA", Tensor(Float(), (4, 2)))
    wB = submodule.add_input_value("wB", Tensor(Float(), (2, 1)))
    y = submodule.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["y"])
    z = submodule.add_op("MatMul", "MatMul1", inputs=[y, wB], output_names=["z"])

    zis = module.add_op(
        "Pmap",
        inputs=[xs, wAs, wBs],
        attributes={"devices": [d0, d1]},
        submodules=[submodule],
        output_names=["zis"],
    )

    infer_shapes(module)

    print(module)

    # TODO: Verify submodule shapes and devices

    assert zis.type.types[0].shape == (8, 1)
    assert zis.type.types[0].device == d0
    assert zis.type.types[1].shape == (8, 1)
    assert zis.type.types[1].device == d1


def test_scatter():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = module.add_input_value("x", Tensor(Float(), (4, 4)))
    xs = module.add_op(
        "Scatter",
        "Scatter/x",
        inputs=[x],
        attributes={"split_dim": 0, "devices": [d0, d1]},
        output_names=["xs"],
    )
    infer_shapes(module)

    assert isinstance(xs.type, ValueTuple)
    assert xs.type.types[0].shape == (2, 4)
    assert xs.type.types[0].device == d0
    assert xs.type.types[1].shape == (2, 4)
    assert xs.type.types[1].device == d1
