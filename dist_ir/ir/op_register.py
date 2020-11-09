from .type import Tensor
from .value import Value


class OpRegisterEntry:
    def __init__(self, input_types, output_types):
        self._input_types = input_types
        self._output_types = output_types

    def infer_types(self, op_name, inputs):
        if len(inputs) != len(self._input_types):
            raise ValueError(
                f"Expected {len(self._input_types)} inputs, got {len(input_types)}"
            )
        for i, (input, input_type) in enumerate(zip(inputs, self._input_types)):
            if not isinstance(input.type, input_type):
                raise ValueError(
                    f"Expected input of type {input_type} for input {i}, got input of type {type(input)}"
                )
        output_values = []
        for i, output_type in enumerate(self._output_types):
            output_name = f"{op_name}/{i}"
            output_values.append(Value(output_name, type=output_type()))
        return output_values


# TODO: Add cost inference functions
OpRegister = {
    "Add": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "MatMul": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
}
