import unittest
import dist_ir
import numpy as np
import torch
import sys

class TestMatMul(unittest.TestCase):

    def __init__(self, testName, backend):
        super(TestMatMul, self).__init__(testName)
        self._backend = backend

    def setUp(self):
        if self._backend == 'numpy':
            self._t1 = dist_ir.ops.Tensor(data=np.random.normal(size=(4, 4)))
            self._t2 = dist_ir.ops.Tensor(data=np.random.normal(size=(4, 4)))
        elif self._backend == 'torch':
            self._t1 = dist_ir.ops.Tensor(data=torch.randn(size=(4, 4)))
            self._t2 = dist_ir.ops.Tensor(data=torch.randn(size=(4, 4)))
        else:
            raise ValueError(f'Unknown backend {self._backend}')
        print(f'Backend: {self._backend}')

    def test_single_matmul(self):
        my_graph = dist_ir.graph.Graph(backend=self._backend)
        my_graph.add_node('matmul')

        result = my_graph.compute(self._t1, self._t2)
        if self._backend == 'numpy':
            self.assertTrue(np.array_equal(result, np.matmul(self._t1.data, self._t2.data)))
        elif self._backend == 'torch':
            self.assertTrue(torch.all(result.eq(torch.matmul(self._t1.data, self._t2.data))))


if __name__=='__main__':
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(TestMatMul)
    suite = unittest.TestSuite()
    for backend in ['numpy', 'torch']:
        for test_name in test_names:
            suite.addTest(TestMatMul(test_name, backend=backend))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
