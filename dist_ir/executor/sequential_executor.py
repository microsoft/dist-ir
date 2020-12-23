from typing import Any, Dict, List

from .backend_register import BackendRegister
from ..ir import Function, Op


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
                inp_names = (e.name for e in op.get_subfunction(0).get_inputs())
                inp_data = {n: v for n, v in zip(inp_names, inps)}
                outs = self.compute(op.get_subfunction(0), inp_data)
                # Match output names to output data using the function output order.
                ordered_outs = [
                    outs[e.name] for e in op.get_subfunction(0).get_outputs()
                ]
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

    def compute(self, function: Function, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the function given the specified inputs and returns the final result.

        Args:
          function: The function to execute.
          input_data: A map from input tensor name to data represented in the
                      specified backend.

        Returns:
          A map from output tensor name to output tensor.
        """
        output_data = {}
        consumers = {}
        ops = function.get_ops()

        # Execute ops in topological order.
        for op_name, op in ops.items():
            inputs = []
            in_edges = op.get_in_edges()
            for in_edge in in_edges:
                input_name = in_edge.name
                if function.is_input(input_name):
                    input_name = in_edge.name
                    if input_name not in input_data:
                        raise ValueError(
                            f"Could not find input {input_name} in input_data"
                        )
                    input_value = input_data[input_name]
                elif input_name in output_data:
                    input_value = output_data[input_name]
                    consumers[input_name] -= 1
                else:
                    raise ValueError(f"Invalid input {input_name} for op {op_name}")
                inputs.append(input_value)

            res = self._compute_op(op, inputs)
            out_edges = op.get_out_edges()
            for i, out_edge in enumerate(out_edges):
                output_data[out_edge.name] = res[i]
                consumers[out_edge.name] = len(
                    function.get_consumers_for_value(out_edge.name)
                )

            # Garbage collect the fully consumed output tensors.
            to_free = []
            for output_name in output_data:
                if consumers[output_name] == 0 and not function.is_output(output_name):
                    to_free.append(output_name)
            for output_name in to_free:
                del output_data[output_name]

        # Return the outputs.
        return output_data
