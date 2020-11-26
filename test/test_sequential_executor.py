import numpy as np
import pytest
import sys
import torch

from dist_ir.ir import Module
from dist_ir.ir.type import Float, Tensor
from dist_ir.executor import SequentialExecutor


class Helper:
    def __init__(self, backend):
        self._backend = backend
        self._executor = SequentialExecutor(self._backend)
        self._module = Module()
        if self._backend == "numpy":
            a = np.random.normal(size=(4, 4))
            b = np.random.normal(size=(4, 4))
            c = np.random.normal(size=(4, 4))
            self._t1 = self._module.add_input_value("a", Tensor(Float(), (4, 4)))
            self._t2 = self._module.add_input_value("b", Tensor(Float(), (4, 4)))
            self._t3 = self._module.add_input_value("c", Tensor(Float(), (4, 4)))
        elif self._backend == "torch":
            a = torch.randn(size=(4, 4))
            b = torch.randn(size=(4, 4))
            c = torch.randn(size=(4, 4))
            self._t1 = self._module.add_input_value("a", Tensor(Float(), (4, 4)))
            self._t2 = self._module.add_input_value("b", Tensor(Float(), (4, 4)))
            self._t3 = self._module.add_input_value("c", Tensor(Float(), (4, 4)))
        else:
            raise ValueError(f"Unknown backend {self._backend}")
        self._input_data = {
            "a": a,
            "b": b,
            "c": c,
        }
        print(f"Backend: {self._backend}")


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    return request.param


def test_single_add(backend):
    h = Helper(backend)
    h._module.add_op("Add", "Add_0", inputs=[h._t1, h._t2])
    output_data = h._executor.compute(h._module, h._input_data)
    result = output_data["Add_0/0"]
    if h._backend == "numpy":
        assert np.array_equal(result, np.add(h._input_data["a"], h._input_data["b"]))
    elif h._backend == "torch":
        assert result.equal(torch.add(h._input_data["a"], h._input_data["b"]))


def test_double_add(backend):
    h = Helper(backend)
    x = h._module.add_op("Add", "Add_0", inputs=[h._t1, h._t2])
    h._module.add_op("Add", "Add_1", inputs=[h._t3, x])
    output_data = h._executor.compute(h._module, h._input_data)
    result = output_data["Add_1/0"]
    if h._backend == "numpy":
        assert np.array_equal(
            result,
            np.add(h._input_data["c"], np.add(h._input_data["a"], h._input_data["b"])),
        )
    elif h._backend == "torch":
        assert result.equal(
            torch.add(
                h._input_data["c"],
                torch.add(h._input_data["a"], h._input_data["b"]),
            )
        )


def test_double_add_inverted(backend):
    h = Helper(backend)
    x = h._module.add_op("Add", "Add_0", inputs=[h._t1, h._t2])
    h._module.add_op("Add", "Add_1", inputs=[x, h._t3])
    output_data = h._executor.compute(h._module, h._input_data)
    result = output_data["Add_1/0"]
    if h._backend == "numpy":
        assert np.array_equal(
            result,
            np.add(np.add(h._input_data["a"], h._input_data["b"]), h._input_data["c"]),
        )
    elif h._backend == "torch":
        assert result.equal(
            torch.add(
                torch.add(h._input_data["a"], h._input_data["b"]),
                h._input_data["c"],
            )
        )


def test_single_matmul(backend):
    h = Helper(backend)
    h._module.add_op("MatMul", "MatMul_0", inputs=[h._t1, h._t2])
    output_data = h._executor.compute(h._module, h._input_data)
    result = output_data["MatMul_0/0"]
    if h._backend == "numpy":
        assert np.array_equal(result, np.matmul(h._input_data["a"], h._input_data["b"]))
    elif h._backend == "torch":
        assert result.equal(torch.matmul(h._input_data["a"], h._input_data["b"]))


def test_double_matmul(backend):
    h = Helper(backend)
    x = h._module.add_op("MatMul", "MatMul_0", inputs=[h._t1, h._t2])
    h._module.add_op("MatMul", "MatMul_1", inputs=[h._t3, x])
    output_data = h._executor.compute(h._module, h._input_data)
    result = output_data["MatMul_1/0"]
    if h._backend == "numpy":
        assert np.array_equal(
            result,
            np.matmul(
                h._input_data["c"], np.matmul(h._input_data["a"], h._input_data["b"])
            ),
        )
    elif h._backend == "torch":
        assert result.equal(
            torch.matmul(
                h._input_data["c"],
                torch.matmul(h._input_data["a"], h._input_data["b"]),
            )
        )


def test_double_matmul_inverted(backend):
    h = Helper(backend)
    x = h._module.add_op("MatMul", "MatMul_0", inputs=[h._t1, h._t2])
    h._module.add_op("MatMul", "MatMul_1", inputs=[x, h._t3])
    output_data = h._executor.compute(h._module, h._input_data)
    result = output_data["MatMul_1/0"]
    if h._backend == "numpy":
        assert np.array_equal(
            result,
            np.matmul(
                np.matmul(h._input_data["a"], h._input_data["b"]), h._input_data["c"]
            ),
        )
    elif h._backend == "torch":
        assert result.equal(
            torch.matmul(
                torch.matmul(h._input_data["a"], h._input_data["b"]),
                h._input_data["c"],
            )
        )


# TODO: Add test for op with multiple outputs
