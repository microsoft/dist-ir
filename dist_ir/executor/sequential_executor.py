from typing import Any, Sequence, Tuple

from .absint import AbstractInterpreter, convert_impls_to_semantics
from .backend_register import BackendRegister
from ..ir import Function, Op, Value


class SequentialExecutor:
    def __init__(self, backend):
        if backend not in BackendRegister:
            raise ValueError(f"Unknown backend {backend}")
        semantics = convert_impls_to_semantics(BackendRegister[backend])
        self.interpreter = AbstractInterpreter(semantics=semantics)

    def compute(self, function: Function, inputs: Sequence[Any]) -> Tuple[Any]:
        """Executes the function given the specified inputs and returns the final result.

        Args:
          function: The function to execute.
          inputs: A sequence of input data represented in the specified backend.

        Returns:
          A tuple of outputs.
        """
        state = self.interpreter.interpret(function, inputs)
        return tuple(state.env[v] for v in function.outputs)
