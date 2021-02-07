from dataclasses import dataclass


@dataclass(frozen=True)
class OpRegisterEntry:
    num_inputs: int = 0
    num_outputs: int = 0
    variadic_inputs: bool = False
    variadic_outputs: bool = False


OpRegister = {
    "Add": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Allreduce": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "BiasFastGeluGrad_dX": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "BiasDropout": OpRegisterEntry(num_inputs=5, num_outputs=2),
    "BiasSoftmax": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Broadcast": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "BroadcastGradientArgs": OpRegisterEntry(num_inputs=2, num_outputs=2),
    "Cast": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Concat": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Div": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Dropout": OpRegisterEntry(num_inputs=3, num_outputs=2),
    "DropoutGrad": OpRegisterEntry(num_inputs=4, num_outputs=1),
    "Expand": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Gather": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "GatherGrad": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "GatherND": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "GatherNDGrad": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "Gemm": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "Group": OpRegisterEntry(variadic_inputs=True, num_outputs=1),
    "FastGelu": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "FusedMatMul": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "InPlaceAccumulator": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Identity": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Join": OpRegisterEntry(variadic_inputs=True, num_outputs=1),
    "LambOptimizer": OpRegisterEntry(variadic_inputs=True, variadic_outputs=True),
    "LayerNormalization": OpRegisterEntry(num_inputs=3, num_outputs=3),
    "LayerNormalizationGrad": OpRegisterEntry(num_inputs=5, num_outputs=3),
    "Loss": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "LossGrad": OpRegisterEntry(num_inputs=2, num_outputs=1),
    # TODO support variadic number of inputs
    "Min": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "MatMul": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "MatMulGrad": OpRegisterEntry(num_inputs=3, num_outputs=2),
    "Min": OpRegisterEntry(variadic_inputs=True, num_outputs=1),
    "MixedPrecisionScale": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "MPIGather": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "MPIReduce": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Mul": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Opt": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "ReduceAllL2": OpRegisterEntry(variadic_inputs=True, num_outputs=1),
    "ReduceSum": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "ReduceSumTraining": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Relu": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "ReluGrad": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Reshape": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Scatter": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Select": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Send": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "SGDOptimizer": OpRegisterEntry(num_inputs=3, num_outputs=2),
    "Shape": OpRegisterEntry(num_inputs=1, num_outputs=1),
    # TODO allow optional inputs for things like slice
    "Slice": OpRegisterEntry(num_inputs=4, num_outputs=1),
    "Softmax": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "SoftmaxGrad": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "SoftmaxCrossEntropy": OpRegisterEntry(num_inputs=2, num_outputs=2),
    "SoftmaxCrossEntropyGrad": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "SoftmaxCrossEntropyLoss": OpRegisterEntry(num_inputs=2, num_outputs=2),
    "SoftmaxCrossEntropyLossGrad": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "Split": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Sub": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Sum": OpRegisterEntry(variadic_inputs=True, num_outputs=1),
    "Unsqueeze": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Tanh": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Transpose": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "ZeroGradient": OpRegisterEntry(num_inputs=2, num_outputs=1),
}
