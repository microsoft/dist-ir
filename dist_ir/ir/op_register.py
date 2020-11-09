from .type import Tensor
from .value import Value


class OpRegisterEntry:
    def __init__(self, input_types, output_types):
        self._input_types = input_types
        self._output_types = output_types

    def infer_types(self, op_name, inputs):
        if len(inputs) != len(self._input_types):
            raise ValueError(
                f"Op {op_name}: Expected {len(self._input_types)} inputs, got {len(inputs)}"
            )
        for i, (input, input_type) in enumerate(zip(inputs, self._input_types)):
            if not isinstance(input.type, input_type):
                raise ValueError(
                    f"Op {op_name}: Expected input of type {input_type} for input {i}, got input of type {type(input)}"
                )
        output_values = []
        for i, output_type in enumerate(self._output_types):
            output_name = f"{op_name}/{i}"
            output_values.append(Value(output_name, type=output_type()))
        return output_values


OpRegister = {
    "Add": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "BroadcastGradientArgs": OpRegisterEntry(
        input_types=[Tensor, Tensor], output_types=[Tensor]
    ),
    "Gemm": OpRegisterEntry(
        input_types=[Tensor, Tensor, Tensor], output_types=[Tensor]
    ),
    "MatMul": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "ReduceSumTraining": OpRegisterEntry(
        input_types=[Tensor, Tensor], output_types=[Tensor]
    ),
    "Relu": OpRegisterEntry(input_types=[Tensor], output_types=[Tensor]),
    "ReluGrad": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Reshape": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "SGDOptimizer": OpRegisterEntry(
        input_types=[Tensor, Tensor, Tensor], output_types=[Tensor]
    ),
    "Shape": OpRegisterEntry(input_types=[Tensor], output_types=[Tensor]),
    "SoftmaxCrossEntropy": OpRegisterEntry(
        input_types=[Tensor, Tensor], output_types=[Tensor]
    ),
    "SoftmaxCrossEntropyGrad": OpRegisterEntry(
        input_types=[Tensor, Tensor, Tensor], output_types=[Tensor]
    ),
}
