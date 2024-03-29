# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass


@dataclass(frozen=True)
class OpRegisterEntry:
    num_inputs: int = 0
    num_outputs: int = 0
    variadic_inputs: bool = False
    variadic_outputs: bool = False


OpRegister = {
    "Add": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Allgather": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "BiasFastGeluGrad_dX": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "BiasDropout": OpRegisterEntry(num_inputs=5, num_outputs=2),
    "BiasSoftmax": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "BroadcastGradientArgs": OpRegisterEntry(num_inputs=2, num_outputs=2),
    "Cast": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Concat": OpRegisterEntry(variadic_inputs=True, num_outputs=1),
    "Constant": OpRegisterEntry(num_inputs=0, num_outputs=1),
    "ConstantOfShape": OpRegisterEntry(num_inputs=1, num_outputs=1),
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
    "Loss": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "LossGrad": OpRegisterEntry(num_inputs=3, num_outputs=1),
    # TODO support variadic number of inputs
    "Min": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "MatMul": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "MatMulGrad": OpRegisterEntry(num_inputs=3, num_outputs=2),
    "Min": OpRegisterEntry(variadic_inputs=True, num_outputs=1),
    "MixedPrecisionScale": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "MPIAllgather": OpRegisterEntry(variadic_inputs=True, variadic_outputs=True),
    "MPIAllreduce": OpRegisterEntry(variadic_inputs=True, variadic_outputs=True),
    "MPIBroadcast": OpRegisterEntry(num_inputs=1, variadic_outputs=True),
    "MPIGather": OpRegisterEntry(variadic_inputs=True, num_outputs=1),
    "MPIReduce": OpRegisterEntry(variadic_inputs=True, num_outputs=1),
    "MPIScatter": OpRegisterEntry(num_inputs=1, variadic_outputs=True),
    "MPIAllreduceFromTupleType": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "MPIBroadcastToTupleType": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "MPIGatherFromTupleType": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "MPIReduceFromTupleType": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "MPIScatterToTupleType": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Mul": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "NonZero": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Opt": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "RecvP2P": OpRegisterEntry(num_inputs=0, num_outputs=1),
    "ReduceAllL2": OpRegisterEntry(variadic_inputs=True, num_outputs=1),
    "ReduceMean": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "ReduceSum": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "ReduceSumTraining": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Relu": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "ReluGrad": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Reshape": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Pow": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Select": OpRegisterEntry(num_inputs=1, num_outputs=1),
    # TODO call the combined one SendRecv?
    "Send": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "SendP2P": OpRegisterEntry(num_inputs=1, num_outputs=0),
    "SGDOptimizer": OpRegisterEntry(variadic_inputs=True, variadic_outputs=True),
    "Shape": OpRegisterEntry(num_inputs=1, num_outputs=1),
    # TODO allow optional inputs for things like slice
    # "Slice": OpRegisterEntry(num_inputs=4, num_outputs=1),
    "Slice": OpRegisterEntry(num_inputs=5, num_outputs=1),
    "Softmax": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "SoftmaxGrad": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "SoftmaxCrossEntropy": OpRegisterEntry(num_inputs=2, num_outputs=2),
    "SoftmaxCrossEntropyGrad": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "SoftmaxCrossEntropyLoss": OpRegisterEntry(num_inputs=2, num_outputs=2),
    "SoftmaxCrossEntropyLossGrad": OpRegisterEntry(num_inputs=3, num_outputs=1),
    # Splits on explicitly specified, potentially non-uniform boundaries
    "Split": OpRegisterEntry(num_inputs=1, variadic_outputs=True),
    # Splits uniformly
    "SplitUniform": OpRegisterEntry(num_inputs=1, variadic_outputs=True),
    "SplitUniformToTupleType": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Sqrt": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Squeeze": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Sub": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Sum": OpRegisterEntry(variadic_inputs=True, num_outputs=1),
    "Unsqueeze": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Tanh": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Transpose": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "ZeroGradient": OpRegisterEntry(num_inputs=2, num_outputs=1),
}
