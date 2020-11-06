import numpy as np
import pytest
import sys
import torch

import dist_ir


class Helper:
    def __init__(self, backend):
        self._backend = backend
        self._executor = dist_ir.executor.SequentialExecutor(self._backend)
        self._graph = dist_ir.graph.Graph()
        if self._backend == "numpy":
            data = np.random.normal(size=(4, 4))
            self._t1 = self._graph.add_input_tensor(name="a")
            self._t2 = self._graph.add_input_tensor(name="b")
            self._t3 = self._graph.add_input_tensor(name="c")
        elif self._backend == "torch":
            data = torch.randn(size=(4, 4))
            self._t1 = self._graph.add_input_tensor(name="a")
            self._t2 = self._graph.add_input_tensor(name="b")
            self._t3 = self._graph.add_input_tensor(name="c")
        else:
            raise ValueError(f"Unknown backend {self._backend}")
        self._input_data = {
            "a": data,
            "b": data,
            "c": data,
        }
        print(f"Backend: {self._backend}")


@pytest.fixture(params=["numpy", "torch"])
def helper_obj(request):
    return Helper(request.param)


def test_single_add(helper_obj):
    h = helper_obj
    h._graph.add_node("Add_0", "Add", h._t1, h._t2)
    outputs = h._executor.compute(h._graph, h._input_data)
    result = outputs["Add_0"].data
    if h._backend == "numpy":
        assert np.array_equal(result, np.add(h._t1.data, h._t2.data))
    elif h._backend == "torch":
        assert torch.all(result.eq(torch.add(h._t1.data, h._t2.data)))


def test_double_add(helper_obj):
    h = helper_obj
    x = h._graph.add_node("Add_0", "Add", h._t1, h._t2)
    h._graph.add_node("Add_1", "Add", h._t3, x)
    outputs = h._executor.compute(h._graph, h._input_data)
    result = outputs["Add_1"].data
    if h._backend == "numpy":
        assert np.array_equal(
            result, np.add(h._t3.data, np.add(h._t1.data, h._t2.data))
        )
    elif h._backend == "torch":
        assert torch.all(
            result.eq(torch.add(h._t3.data, torch.add(h._t1.data, h._t2.data)))
        )


def test_double_add_inverted(helper_obj):
    h = helper_obj
    x = h._graph.add_node("Add_0", "Add", h._t1, h._t2)
    h._graph.add_node("Add_1", "Add", x, h._t3)
    outputs = h._executor.compute(h._graph, h._input_data)
    result = outputs["Add_1"].data
    if h._backend == "numpy":
        assert np.array_equal(
            result, np.add(np.add(h._t1.data, h._t2.data), h._t3.data)
        )
    elif h._backend == "torch":
        assert torch.all(
            result.eq(torch.add(torch.add(h._t1.data, h._t2.data), h._t3.data))
        )


def test_single_matmul(helper_obj):
    h = helper_obj
    h._graph.add_node("MatMul_0", "MatMul", h._t1, h._t2)
    outputs = h._executor.compute(h._graph, h._input_data)
    result = outputs["MatMul_0"].data
    if h._backend == "numpy":
        assert np.array_equal(result, np.matmul(h._t1.data, h._t2.data))
    elif h._backend == "torch":
        assert torch.all(result.eq(torch.matmul(h._t1.data, h._t2.data)))


def test_double_matmul(helper_obj):
    h = helper_obj
    x = h._graph.add_node("MatMul_0", "MatMul", h._t1, h._t2)
    h._graph.add_node("MatMul_1", "MatMul", h._t3, x)
    outputs = h._executor.compute(h._graph, h._input_data)
    result = outputs["MatMul_1"].data
    if h._backend == "numpy":
        assert np.array_equal(
            result,
            np.matmul(h._t3.data, np.matmul(h._t1.data, h._t2.data)),
        )
    elif h._backend == "torch":
        assert torch.all(
            result.eq(torch.matmul(h._t3.data, torch.matmul(h._t1.data, h._t2.data)))
        )


def test_double_matmul_inverted(helper_obj):
    h = helper_obj
    x = h._graph.add_node("MatMul_0", "MatMul", h._t1, h._t2)
    h._graph.add_node("MatMul_1", "MatMul", x, h._t3)
    outputs = h._executor.compute(h._graph, h._input_data)
    result = outputs["MatMul_1"].data
    if h._backend == "numpy":
        assert np.array_equal(
            result,
            np.matmul(np.matmul(h._t1.data, h._t2.data), h._t3.data),
        )
    elif h._backend == "torch":
        assert torch.all(
            result.eq(torch.matmul(torch.matmul(h._t1.data, h._t2.data), h._t3.data))
        )