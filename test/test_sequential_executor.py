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
            data = np.random.normal(size=(4, 4))
            self._t1 = self._module.add_input_value("a", Tensor(Float(), (4, 4)))
            self._t2 = self._module.add_input_value("b", Tensor(Float(), (4, 4)))
            self._t3 = self._module.add_input_value("c", Tensor(Float(), (4, 4)))
        elif self._backend == "torch":
            data = torch.randn(size=(4, 4))
            self._t1 = self._module.add_input_value("a", Tensor(Float(), (4, 4)))
            self._t2 = self._module.add_input_value("b", Tensor(Float(), (4, 4)))
            self._t3 = self._module.add_input_value("c", Tensor(Float(), (4, 4)))
        else:
            raise ValueError(f"Unknown backend {self._backend}")
        self._input_data = {
            "a": data,
            "b": data,
            "c": data,
        }
        print(f"Backend: {self._backend}")


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    return request.param


def test_single_add(backend):
    h = Helper(backend)
    h._module.add_op("Add", "Add_0", [h._t1, h._t2])
    outputs = h._executor.compute(h._module, h._input_data)
    result = outputs["Add_0/0"].data
    if h._backend == "numpy":
        assert np.array_equal(result, np.add(h._t1.data, h._t2.data))
    elif h._backend == "torch":
        assert torch.all(result.eq(torch.add(h._t1.data, h._t2.data)))


def test_double_add(backend):
    h = Helper(backend)
    x = h._module.add_op("Add", "Add_0", [h._t1, h._t2])
    x_outputs = x.get_out_edges()
    h._module.add_op("Add", "Add_1", [h._t3, x_outputs[0]])
    outputs = h._executor.compute(h._module, h._input_data)
    result = outputs["Add_1/0"].data
    if h._backend == "numpy":
        assert np.array_equal(
            result, np.add(h._t3.data, np.add(h._t1.data, h._t2.data))
        )
    elif h._backend == "torch":
        assert torch.all(
            result.eq(torch.add(h._t3.data, torch.add(h._t1.data, h._t2.data)))
        )


def test_double_add_inverted(backend):
    h = Helper(backend)
    x = h._module.add_op("Add", "Add_0", [h._t1, h._t2])
    x_outputs = x.get_out_edges()
    h._module.add_op("Add", "Add_1", [x_outputs[0], h._t3])
    outputs = h._executor.compute(h._module, h._input_data)
    result = outputs["Add_1/0"].data
    if h._backend == "numpy":
        assert np.array_equal(
            result, np.add(np.add(h._t1.data, h._t2.data), h._t3.data)
        )
    elif h._backend == "torch":
        assert torch.all(
            result.eq(torch.add(torch.add(h._t1.data, h._t2.data), h._t3.data))
        )


def test_single_matmul(backend):
    h = Helper(backend)
    h._module.add_op("MatMul", "MatMul_0", [h._t1, h._t2])
    outputs = h._executor.compute(h._module, h._input_data)
    result = outputs["MatMul_0/0"].data
    if h._backend == "numpy":
        assert np.array_equal(result, np.matmul(h._t1.data, h._t2.data))
    elif h._backend == "torch":
        assert torch.all(result.eq(torch.matmul(h._t1.data, h._t2.data)))


def test_double_matmul(backend):
    h = Helper(backend)
    x = h._module.add_op("MatMul", "MatMul_0", [h._t1, h._t2])
    x_outputs = x.get_out_edges()
    h._module.add_op("MatMul", "MatMul_1", [h._t3, x_outputs[0]])
    outputs = h._executor.compute(h._module, h._input_data)
    result = outputs["MatMul_1/0"].data
    if h._backend == "numpy":
        assert np.array_equal(
            result,
            np.matmul(h._t3.data, np.matmul(h._t1.data, h._t2.data)),
        )
    elif h._backend == "torch":
        assert torch.all(
            result.eq(torch.matmul(h._t3.data, torch.matmul(h._t1.data, h._t2.data)))
        )


def test_double_matmul_inverted(backend):
    h = Helper(backend)
    x = h._module.add_op("MatMul", "MatMul_0", [h._t1, h._t2])
    x_outputs = x.get_out_edges()
    h._module.add_op("MatMul", "MatMul_1", [x_outputs[0], h._t3])
    outputs = h._executor.compute(h._module, h._input_data)
    result = outputs["MatMul_1/0"].data
    if h._backend == "numpy":
        assert np.array_equal(
            result,
            np.matmul(np.matmul(h._t1.data, h._t2.data), h._t3.data),
        )
    elif h._backend == "torch":
        assert torch.all(
            result.eq(torch.matmul(torch.matmul(h._t1.data, h._t2.data), h._t3.data))
        )
