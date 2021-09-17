from typing import Any, Sequence, Tuple

from .absint import interpreter
from ..ir import Function


def sequentially_execute(function: Function, inputs: Sequence[Any]) -> Tuple[Any]:
    """Executes the function given the specified inputs and returns the final result.

    Args:
      function: The function to execute.
      inputs: A sequence of input data represented in the specified backend.

    Returns:
      A tuple of outputs.
    """
    state = interpreter.interpret(function, inputs)
    return tuple(state.env[v] for v in function.outputs)
