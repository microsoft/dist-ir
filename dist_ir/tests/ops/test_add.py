import unittest
import dist_ir
import numpy as np
import torch
import sys


class TestAdd(unittest.TestCase):
    def __init__(self, testName, backend):
        super(TestAdd, self).__init__(testName)
        self._backend = backend

    def setUp(self):
        self._graph = dist_ir.graph.Graph(backend=self._backend)
        if self._backend == "numpy":
            data = np.random.normal(size=(4, 4))
            self._t1 = self._graph.add_input(name="a")
            self._t2 = self._graph.add_input(name="b")
            self._t3 = self._graph.add_input(name="c")
        elif self._backend == "torch":
            data = torch.randn(size=(4, 4))
            self._t1 = self._graph.add_input(name="a")
            self._t2 = self._graph.add_input(name="b")
            self._t3 = self._graph.add_input(name="c")
        else:
            raise ValueError(f"Unknown backend {self._backend}")
        self._input_data = {
            "a": data,
            "b": data,
            "c": data,
        }
        print(f"Backend: {self._backend}")

    def test_single_add(self):
        self._graph.add_node("Add", self._t1, self._t2)
        outputs = self._graph.compute(self._input_data)
        result = outputs["Add_0"].data
        if self._backend == "numpy":
            self.assertTrue(
                np.array_equal(result, np.add(self._t1.data, self._t2.data))
            )
        elif self._backend == "torch":
            self.assertTrue(
                torch.all(result.eq(torch.add(self._t1.data, self._t2.data)))
            )

    def test_double_add(self):
        x = self._graph.add_node("Add", self._t1, self._t2)
        self._graph.add_node("Add", self._t3, x)
        outputs = self._graph.compute(self._input_data)
        result = outputs["Add_1"].data
        if self._backend == "numpy":
            self.assertTrue(
                np.array_equal(
                    result, np.add(self._t3.data, np.add(self._t1.data, self._t2.data))
                )
            )
        elif self._backend == "torch":
            self.assertTrue(
                torch.all(
                    result.eq(
                        torch.add(
                            self._t3.data, torch.add(self._t1.data, self._t2.data)
                        )
                    )
                )
            )

    def test_double_add_inverted(self):
        x = self._graph.add_node("Add", self._t1, self._t2)
        self._graph.add_node("Add", x, self._t3)
        outputs = self._graph.compute(self._input_data)
        result = outputs["Add_1"].data
        if self._backend == "numpy":
            self.assertTrue(
                np.array_equal(
                    result, np.add(np.add(self._t1.data, self._t2.data), self._t3.data)
                )
            )
        elif self._backend == "torch":
            self.assertTrue(
                torch.all(
                    result.eq(
                        torch.add(
                            torch.add(self._t1.data, self._t2.data), self._t3.data
                        )
                    )
                )
            )


if __name__ == "__main__":
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(TestAdd)
    suite = unittest.TestSuite()
    for backend in ["numpy", "torch"]:
        for test_name in test_names:
            suite.addTest(TestAdd(test_name, backend=backend))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
