from collections import OrderedDict

import numpy as np
import pytest
import torch

from dist_ir.ir import Device, FunctionMaker, cpprint
from dist_ir.ir.type import Float, Tensor, TupleType
from dist_ir.executor import SequentialExecutor


class Helper:
    def __init__(self, backend):
        self.backend = backend
        self.executor = SequentialExecutor(self.backend)
        self.function = FunctionMaker()
        self.a = self.function.add_input_value("a", Tensor(Float(), (4, 4)))
        self.b = self.function.add_input_value("b", Tensor(Float(), (4, 4)))
        self.c = self.function.add_input_value("c", Tensor(Float(), (4, 4)))
        if self.backend == "numpy":
            a = np.random.normal(size=(4, 4))
            b = np.random.normal(size=(4, 4))
            c = np.random.normal(size=(4, 4))
        elif self.backend == "torch":
            a = torch.randn(size=(4, 4))
            b = torch.randn(size=(4, 4))
            c = torch.randn(size=(4, 4))
        else:
            raise ValueError(f"Unknown backend {self.backend}")
        self.input_data = OrderedDict(((self.a, a), (self.b, b), (self.c, c)))
        print(f"Backend: {self.backend}")


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    return request.param


def test_single_add(backend):
    h = Helper(backend)
    res = h.function.add_op("Add", "Add_0", inputs=[h.a, h.b])
    h.function.set_outputs([res])
    h.function = h.function.finalize()
    (result,) = h.executor.compute(h.function, h.input_data.values())
    if h.backend == "numpy":
        assert np.array_equal(result, np.add(h.input_data[h.a], h.input_data[h.b]))
    elif h.backend == "torch":
        assert result.equal(torch.add(h.input_data[h.a], h.input_data[h.b]))


def test_double_add(backend):
    h = Helper(backend)
    x = h.function.add_op("Add", "Add_0", inputs=[h.a, h.b])
    res = h.function.add_op("Add", "Add_1", inputs=[h.c, x])
    h.function.set_outputs([res])
    h.function = h.function.finalize()
    (result,) = h.executor.compute(h.function, h.input_data.values())
    if h.backend == "numpy":
        assert np.array_equal(
            result,
            np.add(h.input_data[h.c], np.add(h.input_data[h.a], h.input_data[h.b])),
        )
    elif h.backend == "torch":
        assert result.equal(
            torch.add(
                h.input_data[h.c],
                torch.add(h.input_data[h.a], h.input_data[h.b]),
            )
        )


def test_double_add_inverted(backend):
    h = Helper(backend)
    x = h.function.add_op("Add", "Add_0", inputs=[h.a, h.b])
    res = h.function.add_op("Add", "Add_1", inputs=[x, h.c])
    h.function.set_outputs([res])
    h.function = h.function.finalize()
    (result,) = h.executor.compute(h.function, h.input_data.values())
    if h.backend == "numpy":
        assert np.array_equal(
            result,
            np.add(np.add(h.input_data[h.a], h.input_data[h.b]), h.input_data[h.c]),
        )
    elif h.backend == "torch":
        assert result.equal(
            torch.add(
                torch.add(h.input_data[h.a], h.input_data[h.b]),
                h.input_data[h.c],
            )
        )


def test_single_matmul(backend):
    h = Helper(backend)
    res = h.function.add_op("MatMul", "MatMul_0", inputs=[h.a, h.b])
    h.function.set_outputs([res])
    h.function = h.function.finalize()
    (result,) = h.executor.compute(h.function, h.input_data.values())
    if h.backend == "numpy":
        assert np.array_equal(result, np.matmul(h.input_data[h.a], h.input_data[h.b]))
    elif h.backend == "torch":
        assert result.equal(torch.matmul(h.input_data[h.a], h.input_data[h.b]))


def test_double_matmul(backend):
    h = Helper(backend)
    x = h.function.add_op("MatMul", "MatMul_0", inputs=[h.a, h.b])
    res = h.function.add_op("MatMul", "MatMul_1", inputs=[h.c, x])
    h.function.set_outputs([res])
    h.function = h.function.finalize()
    (result,) = h.executor.compute(h.function, h.input_data.values())
    if h.backend == "numpy":
        assert np.array_equal(
            result,
            np.matmul(
                h.input_data[h.c], np.matmul(h.input_data[h.a], h.input_data[h.b])
            ),
        )
    elif h.backend == "torch":
        assert result.equal(
            torch.matmul(
                h.input_data[h.c],
                torch.matmul(h.input_data[h.a], h.input_data[h.b]),
            )
        )


def test_double_matmul_inverted(backend):
    h = Helper(backend)
    x = h.function.add_op("MatMul", "MatMul_0", inputs=[h.a, h.b])
    res = h.function.add_op("MatMul", "MatMul_1", inputs=[x, h.c])
    h.function.set_outputs([res])
    h.function = h.function.finalize()
    (result,) = h.executor.compute(h.function, h.input_data.values())
    if h.backend == "numpy":
        assert np.array_equal(
            result,
            np.matmul(
                np.matmul(h.input_data[h.a], h.input_data[h.b]), h.input_data[h.c]
            ),
        )
    elif h.backend == "torch":
        assert result.equal(
            torch.matmul(
                torch.matmul(h.input_data[h.a], h.input_data[h.b]),
                h.input_data[h.c],
            )
        )


# TODO: Add test for op with multiple outputs

# TODO for all pmap tests, make a FunctionMaker helper function to add pmap
# which also creates the device var and sets the attributes etc appropriately.
# This should also be used by transforms/parsers that create pmap ops.


def test_pmap_on_executor():
    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")
    ex = SequentialExecutor("numpy")

    x_type = lambda d: Tensor(Float(), (8, 4), device=d)
    y_type = lambda d: Tensor(Float(), (4, 2), device=d)

    # Concrete inputs:
    _x = np.arange(16 * 4).reshape((16, 4))
    _x_0, _x_1 = _x[:8], _x[8:]
    _y = np.ones((8, 2))
    _y_0, _y_1 = _y[:4], _y[4:]

    # A pmap with 1 input and 1 output
    function = FunctionMaker()
    xs = function.add_input_value("xs", TupleType((x_type(d0), x_type(d1))))
    subfunction = FunctionMaker()
    x = subfunction.add_input_value("x", x_type(None))
    _ = subfunction.add_op("Add", "Add0", inputs=[x, x], output_names=["z"])
    # subfunction.set_outputs()
    subfunction = subfunction.finalize()
    zis = function.add_op(
        "Pmap",
        inputs=[xs],
        attributes={
            "devices": [d0, d1],
            "device_var": Device.get_new_device_variable("gpu"),
        },
        subfunctions=[subfunction],
        output_names=["zis"],
    )
    function = function.finalize()

    cpprint(function)
    res = ex.compute(function, {xs: (_x_0, _x_1)})
    assert np.array_equal(res[zis][0], _x_0 + _x_0)
    assert np.array_equal(res[zis][1], _x_1 + _x_1)

    # A pmap with 2 inputs and 1 output
    function = FunctionMaker()
    xs = function.add_input_value("xs", TupleType((x_type(d0), x_type(d1))))
    ys = function.add_input_value("ys", TupleType((y_type(d0), y_type(d1))))
    subfunction = FunctionMaker()
    x = subfunction.add_input_value("x", x_type(None))
    y = subfunction.add_input_value("y", y_type(None))
    _ = subfunction.add_op("MatMul", "MatMul0", inputs=[x, y], output_names=["z"])
    subfunction = subfunction.finalize()
    zis = function.add_op(
        "Pmap",
        inputs=[xs, ys],
        attributes={
            "devices": [d0, d1],
            "device_var": Device.get_new_device_variable("gpu"),
        },
        subfunctions=[subfunction],
        output_names=["zis"],
    )
    function = function.finalize()

    cpprint(function)
    res = ex.compute(function, {xs: (_x_0, _x_1), ys: (_y_0, _y_1)})
    assert np.array_equal(res[zis][0], np.matmul(_x_0, _y_0))
    assert np.array_equal(res[zis][1], np.matmul(_x_1, _y_1))

    # A pmap with 2 inputs and 2 outputs
    function = FunctionMaker()
    xs = function.add_input_value("xs", TupleType((x_type(d0), x_type(d1))))
    ys = function.add_input_value("ys", TupleType((y_type(d0), y_type(d1))))
    subfunction = FunctionMaker()
    x = subfunction.add_input_value("x", x_type(None))
    y = subfunction.add_input_value("y", y_type(None))
    _ = subfunction.add_op("Add", "Add0", inputs=[x, x], output_names=["w"])
    _ = subfunction.add_op("MatMul", "MatMul0", inputs=[x, y], output_names=["z"])
    subfunction = subfunction.finalize()
    (wis, zis) = function.add_op(
        "Pmap",
        inputs=[xs, ys],
        attributes={
            "devices": [d0, d1],
            "device_var": Device.get_new_device_variable("gpu"),
        },
        subfunctions=[subfunction],
        output_names=["wis", "zis"],
    )
    function = function.finalize()

    cpprint(function)
    res = ex.compute(function, {xs: (_x_0, _x_1), ys: (_y_0, _y_1)})
    assert np.array_equal(res[wis][0], _x_0 + _x_0)
    assert np.array_equal(res[wis][1], _x_1 + _x_1)
    assert np.array_equal(res[zis][0], np.matmul(_x_0, _y_0))
    assert np.array_equal(res[zis][1], np.matmul(_x_1, _y_1))

    # A pmap with a single device
    function = FunctionMaker()
    xs = function.add_input_value("xs", TupleType((x_type(None),)))
    ys = function.add_input_value("ys", TupleType((y_type(None),)))
    subfunction = FunctionMaker()
    x = subfunction.add_input_value("x", x_type(None))
    y = subfunction.add_input_value("y", y_type(None))
    _ = subfunction.add_op("Add", "Add0", inputs=[x, x], output_names=["w"])
    _ = subfunction.add_op("MatMul", "MatMul0", inputs=[x, y], output_names=["z"])
    subfunction = subfunction.finalize()
    (wis, zis) = function.add_op(
        "Pmap",
        inputs=[xs, ys],
        attributes={
            "devices": [d0],
            "device_var": Device.get_new_device_variable("gpu"),
        },
        subfunctions=[subfunction],
        output_names=["wis", "zis"],
    )
    function = function.finalize()

    cpprint(function)
    res = ex.compute(function, {xs: (_x_0,), ys: (_y_0,)})
    assert np.array_equal(res[wis][0], _x_0 + _x_0)
    assert np.array_equal(res[zis][0], np.matmul(_x_0, _y_0))


def test_pmap_dp():
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
    x = subfunction.add_input_value("x", Tensor(Float(), (8, 4)))
    wA = subfunction.add_input_value("wA", Tensor(Float(), (4, 2)))
    wB = subfunction.add_input_value("wB", Tensor(Float(), (2, 1)))
    y = subfunction.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["y"])
    _ = subfunction.add_op("MatMul", "MatMul1", inputs=[y, wB], output_names=["z"])
    subfunction = subfunction.finalize()
    zis = function.add_op(
        "Pmap",
        inputs=[xs, wAs, wBs],
        attributes={
            "devices": [d0, d1],
            "device_var": Device.get_new_device_variable("gpu"),
        },
        subfunctions=[subfunction],
        output_names=["zis"],
    )
    function = function.finalize()
    cpprint(function)

    ex = SequentialExecutor("numpy")
    _x = np.arange(16 * 4).reshape((16, 4))
    x_0, x_1 = _x[:8], _x[8:]
    _wA = np.ones((4, 2))
    _wB = np.ones((2, 1))
    res = ex.compute(
        function,
        {xs: (x_0, x_1), wAs: (_wA, _wA), wBs: (_wB, _wB)},
    )
    assert np.array_equal(res[zis][0], np.matmul(np.matmul(x_0, _wA), _wB))
    assert np.array_equal(res[zis][1], np.matmul(np.matmul(x_1, _wA), _wB))
