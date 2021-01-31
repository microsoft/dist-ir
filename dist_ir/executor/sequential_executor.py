from typing import Any, Dict, List, Sequence

from .absint import AbstractInterpreter, convert_impls_to_semantics
from .backend_register import BackendRegister
from ..ir import Function, Op, Value


class SequentialExecutor:
    def __init__(self, backend):
        if backend not in BackendRegister:
            raise ValueError(f"Unknown backend {backend}")
        semantics = convert_impls_to_semantics(BackendRegister[backend])
        self.interpreter = AbstractInterpreter(semantics=semantics)

    # TODO pmap in absint
    def _compute_op(self, op: Op, inputs: List[Any]):
        """Executes the given op and returns its outputs."""
        op_type = op.op_type
        if op_type == "Pmap":
            # Zip the inputs so that we map over each corresponding value
            inputs = zip(*inputs)
            # Iterate over the inputs
            results = []
            for inps in inputs:
                # Execute subfunction with appropriate inputs
                inp_data = {k: v for k, v in zip(op.subfunctions[0].inputs, inps)}
                outs = self.compute(op.subfunctions[0], inp_data)
                # Match output names to output data using the function output order.
                ordered_outs = [outs[e] for e in op.subfunctions[0].outputs]
                results.append(ordered_outs)
            # Unzip the results
            results = tuple(zip(*results))
            return results
        if op_type not in BackendRegister[self._backend]:
            raise NotImplementedError(
                f"No {self._backend} implementation found for op {op_type}"
            )
        impl = BackendRegister[self._backend][op_type]
        output_data = impl(op, inputs)
        if not isinstance(output_data, tuple):
            output_data = (output_data,)
        return output_data

    def compute(self, function: Function, inputs: Sequence[Any]) -> Dict[Value, Any]:
        """Executes the function given the specified inputs and returns the final result.

        Args:
          function: The function to execute.
          inputs: A sequence of input data represented in the specified backend.

        Returns:
          A map from output value to output data.
        """
        state = self.interpreter.interpret(function, inputs)
        return tuple(state.env[v] for v in function.outputs)
