import pytest

from dist_ir.ir import cpprint, Device, Function, FunctionMaker, Op, Value
from dist_ir.executor.type_inference import infer_types
from dist_ir.ir.type import Float, Tensor, TupleType


def test_add_valid():
    function = FunctionMaker()

    a = function.add_input_value("a", Tensor(Float(), (4, 4)))
    b = function.add_input_value("b", Tensor(Float(), (4, 4)))
    x = function.add_op("Add", "Add0", inputs=[a, b], output_names=["x"])
    function = function.finalize()
    typed_function = infer_types(function, [a, b])
    assert typed_function.outputs[0].type.shape == (4, 4)


def test_add_invalid():
    function = FunctionMaker()

    a = function.add_input_value("a", Tensor(Float(), (8, 4)))
    b = function.add_input_value("b", Tensor(Float(), (4, 2)))
    x = function.add_op("Add", "Add0", inputs=[a, b], output_names=["x"])
    function = function.finalize()
    with pytest.raises(ValueError):
        infer_types(function, [a, b])


def test_allreduce():
    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    xis = Value(
        "xis",
        TupleType(
            (Tensor(Float(), (4, 4), device=d0), Tensor(Float(), (4, 4), device=d1))
        ),
    )
    op1 = Op(
        "Allreduce",
        "Allreduces/xis",
        inputs=[xis],
        output_names=["xs"],
    )
    function = Function("foo", (op1,), (xis,), (op1.outputs[0],))
    function = infer_types(function, [xis])
    xs = function.outputs[0]

    assert isinstance(xs.type, TupleType)
    for i, value_type in enumerate(xis.type.types):
        assert value_type.shape == xs.type.types[i].shape
        assert value_type.device == xs.type.types[i].device


def test_broadcast():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = function.add_input_value("x", Tensor(Float(), (4, 4)))
    xs = function.add_op(
        "Broadcast",
        "Broadcast/x",
        inputs=[x],
        attributes={"devices": [d0, d1]},
        output_names=["xs"],
    )
    function = function.finalize()
    function = infer_types(function, [x])
    xs = function.outputs[0]

    assert isinstance(xs.type, TupleType)
    assert xs.type.types[0].shape == (4, 4)
    assert xs.type.types[0].device == d0
    assert xs.type.types[1].shape == (4, 4)
    assert xs.type.types[1].device == d1


def test_matmul_valid():
    function = FunctionMaker()

    a = function.add_input_value("a", Tensor(Float(), (8, 4)))
    b = function.add_input_value("b", Tensor(Float(), (4, 2)))
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    function = function.finalize()
    function = infer_types(function, [a, b])
    assert function.outputs[0].type.shape == (8, 2)


def test_matmul_invalid():
    function = FunctionMaker()

    a = function.add_input_value("a", Tensor(Float(), (8, 8)))
    b = function.add_input_value("b", Tensor(Float(), (4, 2)))
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    function = function.finalize()
    with pytest.raises(ValueError):
        function = infer_types(function, [a, b])


def test_matmul_grad():
    function = FunctionMaker()

    x = function.add_input_value("x", Tensor(Float(), (8, 4)))
    w = function.add_input_value("w", Tensor(Float(), (4, 2)))
    l = function.add_input_value("l", Tensor(Float(), (8,)))
    dx, dw = function.add_op(
        "MatMulGrad", "MatMulGrad0", inputs=[x, w, l], output_names=["dx", "dw"]
    )
    function = function.finalize()
    function = infer_types(function, [x, w, l])
    dx, dw = function.outputs
    assert dx.type.shape == x.type.shape
    assert dw.type.shape == w.type.shape


def test_pmap():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    xs = function.add_input_value(
        "xs",
        TupleType(
            (Tensor(Float(), (8, 4), device=d0), Tensor(Float(), (8, 4), device=d1))
        ),
    )
    wAs = function.add_input_value(
        "wAs",
        TupleType(
            (Tensor(Float(), (4, 2), device=d0), Tensor(Float(), (4, 2), device=d1))
        ),
    )
    wBs = function.add_input_value(
        "wBs",
        TupleType(
            (Tensor(Float(), (2, 1), device=d0), Tensor(Float(), (2, 1), device=d1))
        ),
    )

    subfunction = FunctionMaker()
    x = subfunction.add_input_value("x", None)
    wA = subfunction.add_input_value("wA", None)
    wB = subfunction.add_input_value("wB", None)
    y = subfunction.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["y"])
    z = subfunction.add_op("MatMul", "MatMul1", inputs=[y, wB], output_names=["z"])
    subfunction = subfunction.finalize()

    zis = function.add_op(
        "Pmap",
        inputs=[xs, wAs, wBs],
        attributes={
            "devices": [d0, d1],
            "device_var": Device.get_new_device_variable(
                "gpu"
            ),  # TODO where best to do this?
        },
        subfunctions=[subfunction],
        output_names=["zis"],
    )

    function = function.finalize()
    cpprint(function)
    function = infer_types(function, [xs, wAs, wBs])
    cpprint(function)

    # TODO: Verify subfunction shapes and devices

    zis = function.outputs[0]
    assert zis.type.types[0].shape == (8, 1)
    assert zis.type.types[0].device == d0
    assert zis.type.types[1].shape == (8, 1)
    assert zis.type.types[1].device == d1


def test_scatter():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = function.add_input_value("x", Tensor(Float(), (4, 4)))
    xs = function.add_op(
        "Scatter",
        "Scatter/x",
        inputs=[x],
        attributes={"dim": 0, "devices": [d0, d1]},
        output_names=["xs"],
    )
    function = function.finalize()
    function = infer_types(function, [x])
    xs = function.outputs[0]

    assert isinstance(xs.type, TupleType)
    assert xs.type.types[0].shape == (2, 4)
    assert xs.type.types[0].device == d0
    assert xs.type.types[1].shape == (2, 4)
    assert xs.type.types[1].device == d1


if __name__ == "__main__":
    test_pmap()
