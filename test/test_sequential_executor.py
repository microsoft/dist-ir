import numpy as np
import pytest
import torch

from dist_ir.ir import Module
from dist_ir.ir.device import Device
from dist_ir.ir.type import Float, Tensor, TupleType
from dist_ir.executor import SequentialExecutor
from dist_ir.executor.shape_inference import infer_shapes


class Helper:
    def __init__(self, backend):
        self.backend = backend
        self.executor = SequentialExecutor(self.backend)
        self.module = Module()
        self.t1 = self.module.add_input_value("a", Tensor(Float(), (4, 4)))
        self.t2 = self.module.add_input_value("b", Tensor(Float(), (4, 4)))
        self.t3 = self.module.add_input_value("c", Tensor(Float(), (4, 4)))
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
        self.input_data = {
            "a": a,
            "b": b,
            "c": c,
        }
        print(f"Backend: {self.backend}")


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    return request.param


def test_single_add(backend):
    h = Helper(backend)
    h.module.add_op("Add", "Add_0", inputs=[h.t1, h.t2])
    output_data = h.executor.compute(h.module, h.input_data)
    result = output_data["Add_0/0"]
    if h.backend == "numpy":
        assert np.array_equal(result, np.add(h.input_data["a"], h.input_data["b"]))
    elif h.backend == "torch":
        assert result.equal(torch.add(h.input_data["a"], h.input_data["b"]))


def test_double_add(backend):
    h = Helper(backend)
    x = h.module.add_op("Add", "Add_0", inputs=[h.t1, h.t2])
    h.module.add_op("Add", "Add_1", inputs=[h.t3, x])
    output_data = h.executor.compute(h.module, h.input_data)
    result = output_data["Add_1/0"]
    if h.backend == "numpy":
        assert np.array_equal(
            result,
            np.add(h.input_data["c"], np.add(h.input_data["a"], h.input_data["b"])),
        )
    elif h.backend == "torch":
        assert result.equal(
            torch.add(
                h.input_data["c"],
                torch.add(h.input_data["a"], h.input_data["b"]),
            )
        )


def test_double_add_inverted(backend):
    h = Helper(backend)
    x = h.module.add_op("Add", "Add_0", inputs=[h.t1, h.t2])
    h.module.add_op("Add", "Add_1", inputs=[x, h.t3])
    output_data = h.executor.compute(h.module, h.input_data)
    result = output_data["Add_1/0"]
    if h.backend == "numpy":
        assert np.array_equal(
            result,
            np.add(np.add(h.input_data["a"], h.input_data["b"]), h.input_data["c"]),
        )
    elif h.backend == "torch":
        assert result.equal(
            torch.add(
                torch.add(h.input_data["a"], h.input_data["b"]),
                h.input_data["c"],
            )
        )


def test_single_matmul(backend):
    h = Helper(backend)
    h.module.add_op("MatMul", "MatMul_0", inputs=[h.t1, h.t2])
    output_data = h.executor.compute(h.module, h.input_data)
    result = output_data["MatMul_0/0"]
    if h.backend == "numpy":
        assert np.array_equal(result, np.matmul(h.input_data["a"], h.input_data["b"]))
    elif h.backend == "torch":
        assert result.equal(torch.matmul(h.input_data["a"], h.input_data["b"]))


def test_double_matmul(backend):
    h = Helper(backend)
    x = h.module.add_op("MatMul", "MatMul_0", inputs=[h.t1, h.t2])
    h.module.add_op("MatMul", "MatMul_1", inputs=[h.t3, x])
    output_data = h.executor.compute(h.module, h.input_data)
    result = output_data["MatMul_1/0"]
    if h.backend == "numpy":
        assert np.array_equal(
            result,
            np.matmul(
                h.input_data["c"], np.matmul(h.input_data["a"], h.input_data["b"])
            ),
        )
    elif h.backend == "torch":
        assert result.equal(
            torch.matmul(
                h.input_data["c"],
                torch.matmul(h.input_data["a"], h.input_data["b"]),
            )
        )


def test_double_matmul_inverted(backend):
    h = Helper(backend)
    x = h.module.add_op("MatMul", "MatMul_0", inputs=[h.t1, h.t2])
    h.module.add_op("MatMul", "MatMul_1", inputs=[x, h.t3])
    output_data = h.executor.compute(h.module, h.input_data)
    result = output_data["MatMul_1/0"]
    if h.backend == "numpy":
        assert np.array_equal(
            result,
            np.matmul(
                np.matmul(h.input_data["a"], h.input_data["b"]), h.input_data["c"]
            ),
        )
    elif h.backend == "torch":
        assert result.equal(
            torch.matmul(
                torch.matmul(h.input_data["a"], h.input_data["b"]),
                h.input_data["c"],
            )
        )


# TODO: Add test for op with multiple outputs


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
    module = Module()
    xs = module.add_input_value("xs", TupleType((x_type(d0), x_type(d1))))
    submodule = Module()
    x = submodule.add_input_value("x", x_type(None))
    _ = submodule.add_op("Add", "Add0", inputs=[x, x], output_names=["z"])
    _ = module.add_op(
        "Pmap",
        inputs=[xs],
        attributes={"devices": [d0, d1]},
        submodules=[submodule],
        output_names=["zis"],
    )
    infer_shapes(module)

    res = ex.compute(module, {"xs": (_x_0, _x_1)})
    assert np.array_equal(res["zis"][0], _x_0 + _x_0)
    assert np.array_equal(res["zis"][1], _x_1 + _x_1)

    # A pmap with 2 inputs and 1 output
    module = Module()
    xs = module.add_input_value("xs", TupleType((x_type(d0), x_type(d1))))
    ys = module.add_input_value("ys", TupleType((y_type(d0), y_type(d1))))
    submodule = Module()
    x = submodule.add_input_value("x", x_type(None))
    y = submodule.add_input_value("y", y_type(None))
    _ = submodule.add_op("MatMul", "MatMul0", inputs=[x, y], output_names=["z"])
    _ = module.add_op(
        "Pmap",
        inputs=[xs, ys],
        attributes={"devices": [d0, d1]},
        submodules=[submodule],
        output_names=["zis"],
    )
    infer_shapes(module)

    res = ex.compute(module, {"xs": (_x_0, _x_1), "ys": (_y_0, _y_1)})
    assert np.array_equal(res["zis"][0], np.matmul(_x_0, _y_0))
    assert np.array_equal(res["zis"][1], np.matmul(_x_1, _y_1))

    # A pmap with 2 inputs and 2 outputs
    module = Module()
    xs = module.add_input_value("xs", TupleType((x_type(d0), x_type(d1))))
    ys = module.add_input_value("ys", TupleType((y_type(d0), y_type(d1))))
    submodule = Module()
    x = submodule.add_input_value("x", x_type(None))
    y = submodule.add_input_value("y", y_type(None))
    _ = submodule.add_op("Add", "Add0", inputs=[x, x], output_names=["w"])
    _ = submodule.add_op("MatMul", "MatMul0", inputs=[x, y], output_names=["z"])
    _ = module.add_op(
        "Pmap",
        inputs=[xs, ys],
        attributes={"devices": [d0, d1]},
        submodules=[submodule],
        output_names=["zis", "wis"],
    )
    infer_shapes(module)

    res = ex.compute(module, {"xs": (_x_0, _x_1), "ys": (_y_0, _y_1)})
    assert np.array_equal(res["wis"][0], _x_0 + _x_0)
    assert np.array_equal(res["wis"][1], _x_1 + _x_1)
    assert np.array_equal(res["zis"][0], np.matmul(_x_0, _y_0))
    assert np.array_equal(res["zis"][1], np.matmul(_x_1, _y_1))

    # TODO Test pmap with only one device


def test_pmap_dp():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    xs = module.add_input_value(
        "xs",
        TupleType(
            (Tensor(Float(), (8, 4), device=d0), Tensor(Float(), (8, 4), device=d1))
        ),
    )
    wAs = module.add_input_value(
        "wAs",
        TupleType(
            (Tensor(Float(), (4, 2), device=d0), Tensor(Float(), (4, 2), device=d1))
        ),
    )
    wBs = module.add_input_value(
        "wBs",
        TupleType(
            (Tensor(Float(), (2, 1), device=d0), Tensor(Float(), (2, 1), device=d1))
        ),
    )

    submodule = Module()
    x = submodule.add_input_value("x", Tensor(Float(), (8, 4)))
    wA = submodule.add_input_value("wA", Tensor(Float(), (4, 2)))
    wB = submodule.add_input_value("wB", Tensor(Float(), (2, 1)))
    y = submodule.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["y"])
    _ = submodule.add_op("MatMul", "MatMul1", inputs=[y, wB], output_names=["z"])
    _ = module.add_op(
        "Pmap",
        inputs=[xs, wAs, wBs],
        attributes={"devices": [d0, d1]},
        submodules=[submodule],
        output_names=["zis"],
    )

    # TODO does this have to be run every time a module is constructed?
    infer_shapes(module)

    ex = SequentialExecutor("numpy")
    _x = np.arange(16 * 4).reshape((16, 4))
    x_0, x_1 = _x[:8], _x[8:]
    _wA = np.ones((4, 2))
    _wB = np.ones((2, 1))
    res = ex.compute(
        module,
        {"xs": (x_0, x_1), "wAs": (_wA, _wA), "wBs": (_wB, _wB)},
    )
    assert np.array_equal(res["zis"][0], np.matmul(np.matmul(x_0, _wA), _wB))
    assert np.array_equal(res["zis"][1], np.matmul(np.matmul(x_1, _wA), _wB))
