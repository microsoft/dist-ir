# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import OrderedDict
from typing import Union

import numpy as np

from dist_ir.executor import ConcreteValue, sequentially_execute
from dist_ir.ir import Device, FunctionMaker, cpprint
from dist_ir.ir.type import Float32, Tensor, TupleType
from dist_ir.ir.value import Value


class Helper:
    def __init__(self):
        self.function = FunctionMaker()
        self.a = self.function.add_input_value("a", Tensor(Float32(), (4, 4)))
        self.b = self.function.add_input_value("b", Tensor(Float32(), (4, 4)))
        self.c = self.function.add_input_value("c", Tensor(Float32(), (4, 4)))
        a = np.random.normal(size=(4, 4))
        b = np.random.normal(size=(4, 4))
        c = np.random.normal(size=(4, 4))
        self.input_data = OrderedDict(((self.a, a), (self.b, b), (self.c, c)))
        for v in self.input_data:
            self.input_data[v] = ConcreteValue(self.input_data[v], None)

    def input(self, v: Value) -> np.ndarray:
        return self.input_data[v].val


def test_single_add():
    h = Helper()
    res = h.function.add_op("Add", "Add_0", inputs=[h.a, h.b])
    h.function.set_outputs([res])
    h.function = h.function.finalize()
    (result,) = sequentially_execute(h.function, h.input_data.values())
    assert np.array_equal(result.val, np.add(h.input(h.a), h.input(h.b)))


def test_double_add():
    h = Helper()
    x = h.function.add_op("Add", "Add_0", inputs=[h.a, h.b])
    res = h.function.add_op("Add", "Add_1", inputs=[h.c, x])
    h.function.set_outputs([res])
    h.function = h.function.finalize()
    (result,) = sequentially_execute(h.function, h.input_data.values())
    assert np.array_equal(
        result.val,
        np.add(h.input(h.c), np.add(h.input(h.a), h.input(h.b))),
    )


def test_double_add_inverted():
    h = Helper()
    x = h.function.add_op("Add", "Add_0", inputs=[h.a, h.b])
    res = h.function.add_op("Add", "Add_1", inputs=[x, h.c])
    h.function.set_outputs([res])
    h.function = h.function.finalize()
    (result,) = sequentially_execute(h.function, h.input_data.values())
    assert np.array_equal(
        result.val,
        np.add(np.add(h.input(h.a), h.input(h.b)), h.input(h.c)),
    )


def test_single_matmul():
    h = Helper()
    res = h.function.add_op("MatMul", "MatMul_0", inputs=[h.a, h.b])
    h.function.set_outputs([res])
    h.function = h.function.finalize()
    (result,) = sequentially_execute(h.function, h.input_data.values())
    assert np.array_equal(result.val, np.matmul(h.input(h.a), h.input(h.b)))


def test_double_matmul():
    h = Helper()
    x = h.function.add_op("MatMul", "MatMul_0", inputs=[h.a, h.b])
    res = h.function.add_op("MatMul", "MatMul_1", inputs=[h.c, x])
    h.function.set_outputs([res])
    h.function = h.function.finalize()
    (result,) = sequentially_execute(h.function, h.input_data.values())
    assert np.array_equal(
        result.val,
        np.matmul(h.input(h.c), np.matmul(h.input(h.a), h.input(h.b))),
    )


def test_double_matmul_inverted():
    h = Helper()
    x = h.function.add_op("MatMul", "MatMul_0", inputs=[h.a, h.b])
    res = h.function.add_op("MatMul", "MatMul_1", inputs=[x, h.c])
    h.function.set_outputs([res])
    h.function = h.function.finalize()
    (result,) = sequentially_execute(h.function, h.input_data.values())
    assert np.array_equal(
        result.val,
        np.matmul(np.matmul(h.input(h.a), h.input(h.b)), h.input(h.c)),
    )


# TODO: Add test for op with multiple outputs

# TODO for all pmap tests, make a FunctionMaker helper function to add pmap
# which also creates the device var and sets the attributes etc appropriately.
# This should also be used by transforms/parsers that create pmap ops.

# TODO pmap tests disabled. If needed, wrap inputs/outputs in ConcreteValues


def _test_pmap_on_executor():
    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x_type = lambda d: Tensor(Float32(), (8, 4), device=d)
    y_type = lambda d: Tensor(Float32(), (4, 2), device=d)

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
    (res,) = sequentially_execute(function, [(_x_0, _x_1)])
    assert np.array_equal(res[0], _x_0 + _x_0)
    assert np.array_equal(res[1], _x_1 + _x_1)

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
    (res,) = sequentially_execute(function, [(_x_0, _x_1), (_y_0, _y_1)])
    assert np.array_equal(res[0], np.matmul(_x_0, _y_0))
    assert np.array_equal(res[1], np.matmul(_x_1, _y_1))

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
    function.set_outputs([wis, zis])
    function = function.finalize()

    cpprint(function)
    (res_wis, res_zis) = sequentially_execute(function, [(_x_0, _x_1), (_y_0, _y_1)])
    assert np.array_equal(res_wis[0], _x_0 + _x_0)
    assert np.array_equal(res_wis[1], _x_1 + _x_1)
    assert np.array_equal(res_zis[0], np.matmul(_x_0, _y_0))
    assert np.array_equal(res_zis[1], np.matmul(_x_1, _y_1))

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
    (res_wis, res_zis) = sequentially_execute(function, [(_x_0,), (_y_0,)])
    assert np.array_equal(res_wis[0], _x_0 + _x_0)
    assert np.array_equal(res_zis[0], np.matmul(_x_0, _y_0))


def _test_pmap_dp():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    xs = function.add_input_value(
        "xs",
        TupleType(
            (Tensor(Float32(), (8, 4), device=d0), Tensor(Float32(), (8, 4), device=d1))
        ),
    )
    wAs = function.add_input_value(
        "wAs",
        TupleType(
            (Tensor(Float32(), (4, 2), device=d0), Tensor(Float32(), (4, 2), device=d1))
        ),
    )
    wBs = function.add_input_value(
        "wBs",
        TupleType(
            (Tensor(Float32(), (2, 1), device=d0), Tensor(Float32(), (2, 1), device=d1))
        ),
    )

    subfunction = FunctionMaker()
    x = subfunction.add_input_value("x", Tensor(Float32(), (8, 4)))
    wA = subfunction.add_input_value("wA", Tensor(Float32(), (4, 2)))
    wB = subfunction.add_input_value("wB", Tensor(Float32(), (2, 1)))
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

    _x = np.arange(16 * 4).reshape((16, 4))
    x_0, x_1 = _x[:8], _x[8:]
    _wA = np.ones((4, 2))
    _wB = np.ones((2, 1))
    (res,) = sequentially_execute(function, [(x_0, x_1), (_wA, _wA), (_wB, _wB)])
    assert np.array_equal(res[0], np.matmul(np.matmul(x_0, _wA), _wB))
    assert np.array_equal(res[1], np.matmul(np.matmul(x_1, _wA), _wB))


if __name__ == "__main__":
    test_single_add("numpy")
