# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Sequence, Tuple

from .absint import interpreter
from ..ir import Function


class SequentialExecutor:
    def __init__(self, backend):
        # TODO remove need for backend
        pass

    def compute(self, function: Function, inputs: Sequence[Any]) -> Tuple[Any]:
        """Executes the function given the specified inputs and returns the final result.

        Args:
          function: The function to execute.
          inputs: A sequence of input data represented in the specified backend.

        Returns:
          A tuple of outputs.
        """
        state = interpreter.interpret(function, inputs)
        return tuple(state.env[v] for v in function.outputs)
