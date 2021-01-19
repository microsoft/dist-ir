from dataclasses import dataclass


@dataclass(frozen=True)
class OpRegisterEntry:
    num_inputs: int
    num_outputs: int


OpRegister = {
    "Add": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Allreduce": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Broadcast": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "BroadcastGradientArgs": OpRegisterEntry(num_inputs=2, num_outputs=2),
    "Concat": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Gather": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Gemm": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "Loss": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "LossGrad": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "MatMul": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "MatMulGrad": OpRegisterEntry(num_inputs=3, num_outputs=2),
    "ReduceSumTraining": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Relu": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "ReluGrad": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Reshape": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Opt": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Scatter": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Select": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Send": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "SGDOptimizer": OpRegisterEntry(num_inputs=3, num_outputs=2),
    "Shape": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "SoftmaxCrossEntropy": OpRegisterEntry(num_inputs=2, num_outputs=2),
    "SoftmaxCrossEntropyGrad": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "Split": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Transpose": OpRegisterEntry(num_inputs=1, num_outputs=1),
}
