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
        self._graph = dist_ir.graph.Graph(backend=self._backend)
        if self._backend == 'numpy':
            self._t1 = self._graph.add_tensor(name='a', data=np.random.normal(size=(4, 4)))
            self._t2 = self._graph.add_tensor(name='b', data=np.random.normal(size=(4, 4)))
            self._t3 = self._graph.add_tensor(name='c', data=np.random.normal(size=(4, 4)))
        elif self._backend == 'torch':
            self._t1 = self._graph.add_tensor(name='a', data=torch.randn(size=(4, 4)))
            self._t2 = self._graph.add_tensor(name='b', data=torch.randn(size=(4, 4)))
            self._t3 = self._graph.add_tensor(name='c', data=torch.randn(size=(4, 4)))
        else:
            raise ValueError(f'Unknown backend {self._backend}')
        print(f'Backend: {self._backend}')

    def test_single_matmul(self):
        self._graph.add_node('matmul')
        outputs = self._graph.compute(self._t1, self._t2)
        result = outputs['matmul_0'].data
        if self._backend == 'numpy':
            self.assertTrue(np.array_equal(result, np.matmul(self._t1.data, self._t2.data)))
        elif self._backend == 'torch':
            self.assertTrue(torch.all(result.eq(torch.matmul(self._t1.data, self._t2.data))))

    def test_double_matmul(self):
        x = self._graph.add_node('matmul')
        self._graph.add_node('matmul', self._t3, x)
        outputs = self._graph.compute(self._t1, self._t2)
        result = outputs['matmul_1'].data
        if self._backend == 'numpy':
            self.assertTrue(np.array_equal(result, np.matmul(self._t3.data, np.matmul(self._t1.data, self._t2.data))))
        elif self._backend == 'torch':
            self.assertTrue(torch.all(result.eq(torch.matmul(self._t3.data, torch.matmul(self._t1.data, self._t2.data)))))

    def test_double_matmul_inverted(self):
        x = self._graph.add_node('matmul')
        self._graph.add_node('matmul', x, self._t3)
        outputs = self._graph.compute(self._t1, self._t2)
        result = outputs['matmul_1'].data
        if self._backend == 'numpy':
            self.assertTrue(np.array_equal(result, np.matmul(np.matmul(self._t1.data, self._t2.data), self._t3.data)))
        elif self._backend == 'torch':
            self.assertTrue(torch.all(result.eq(torch.matmul(torch.matmul(self._t1.data, self._t2.data), self._t3.data))))

if __name__=='__main__':
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(TestMatMul)
    suite = unittest.TestSuite()
    for backend in ['numpy', 'torch']:
        for test_name in test_names:
            suite.addTest(TestMatMul(test_name, backend=backend))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
