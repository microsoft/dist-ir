from .add import Add
from .gemm import GeMM
from .identity import Identity
from .matmul import MatMul
from .relu import ReLU
from .reshape import Reshape
from .shape import Shape
from .softmax_cross_entropy import SoftmaxCrossEntropy

from .training.broadcast_gradient_args import BroadcastGradientArgs
from .training.relu import ReLUGrad
from .training.sgd_optimizer import SGDOptimizer
from .training.softmax_cross_entropy import SoftmaxCrossEntropyGrad
from .training.reduce_sum_training import ReduceSumTraining

OpRegister = {
    "Add": Add,
    "BroadcastGradientArgs": BroadcastGradientArgs,
    "Gemm": GeMM,
    "Identity": Identity,
    "MatMul": MatMul,
    "ReduceSumTraining": ReduceSumTraining,
    "Relu": ReLU,
    "ReluGrad": ReLUGrad,
    "Reshape": Reshape,
    "SGDOptimizer": SGDOptimizer,
    "Shape": Shape,
    "SoftmaxCrossEntropy": SoftmaxCrossEntropy,
    "SoftmaxCrossEntropyGrad": SoftmaxCrossEntropyGrad,
}
