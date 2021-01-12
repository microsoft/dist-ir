from typing import Any, Dict, List

from .backend_register import BackendRegister
from ..ir import Function, Op, Value


class SequentialExecutor:
    def __init__(self, backend):
        if backend not in BackendRegister:
            raise ValueError(f"Unknown backend {backend}")
        self._backend = backend

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

    def compute(
        self, function: Function, input_data: Dict[Value, Any]
    ) -> Dict[Value, Any]:
        """Executes the function given the specified inputs and returns the final result.

        Args:
          function: The function to execute.
          input_data: A map from input value to data represented in the
                      specified backend.

        Returns:
          A map from output value to output data.
        """
        output_data = {}
        consumers = {}

        # Execute ops in topological order.
        for op in function.ops:
            inputs = []
            for in_edge in op.inputs:
                if in_edge in function.inputs:
                    if in_edge not in input_data:
                        raise ValueError(
                            f"Could not find input {in_edge} in input_data"
                        )
                    input_value = input_data[in_edge]
                elif in_edge in output_data:
                    input_value = output_data[in_edge]
                    consumers[in_edge] -= 1
                else:
                    raise ValueError(f"Invalid input {in_edge} for op {op}")
                inputs.append(input_value)

            res = self._compute_op(op, inputs)
            for i, out_edge in enumerate(op.outputs):
                output_data[out_edge] = res[i]
                consumers[out_edge] = len(function.get_consumers(out_edge))

            # Garbage collect the fully consumed output tensors.
            to_free = []
            for out_edge in output_data:
                if consumers[out_edge] == 0 and not out_edge in function.outputs:
                    to_free.append(out_edge)
            for out_edge in to_free:
                del output_data[out_edge]

        # Return the outputs.
        return output_data
