from .backend_register import BackendRegister
from ..ir.type import Tensor


class SequentialExecutor:
    def __init__(self, backend):
        if backend not in BackendRegister:
            raise ValueError(f"Unknown backend {backend}")
        self._backend = backend

    def _resolve_inputs(self, inputs):
        """Converts the given inputs into the form expected by the specified backend."""
        resolved_inputs = []
        for input in inputs:
            # TODO: Support input types beyond Tensor
            if not isinstance(input.type, Tensor):
                raise ValueError(f"Invalid input type {input.type}")
            resolved_inputs.append(input.data)
        return resolved_inputs

    def _compute_op(self, op, inputs):
        """Executes the given op and returns its outputs."""
        op_type = op.op_type
        if op_type not in BackendRegister[self._backend]:
            raise NotImplementedError(
                f"No {self._backend} implementation found for op {op_type}"
            )
        impl = BackendRegister[self._backend][op_type]
        resolved_inputs = self._resolve_inputs(inputs)
        out_edges = op.get_out_edges()
        # TODO: Support multiple output values
        output_data = impl(*resolved_inputs)
        return output_data

    def compute(self, module, input_data):
        """Executes the module given the specified inputs and returns the final result.

        Args:
          module: The module to execute.
          input_data: A map from input tensor name to data represented in the
                      specified backend.

        Returns:
          A map from output tensor name to output tensor.
        """
        output_data = {}
        consumers = {}
        sinks = set()
        ops = module.get_ops()

        # Execute ops in topological order.
        for op_name, op in ops.items():
            inputs = []
            in_edges = op.get_in_edges()
            for in_edge in in_edges:
                input_name = in_edge.name
                if module.is_input(input_name):
                    input_name = in_edge.name
                    if input_name not in input_data:
                        raise ValueError(
                            f"Could not find input {input_name} in input_data"
                        )
                    input_value = module.get_input(input_name)
                    input_value.data = input_data[input_name]
                elif input_name in output_data:
                    input_value = in_edge
                    in_edge.data = output_data[input_name]
                    consumers[input_name] -= 1
                else:
                    raise ValueError(f"Invalid input {input_name} for op {op_name}")
                inputs.append(input_value)

            # TODO: Support more than 1 out edge per op
            out_edges = op.get_out_edges()
            res = self._compute_op(op, inputs)
            output_data[out_edges[0].name] = res
            consumers[out_edges[0].name] = 1

            # TODO: Garbage collect the fully consumed output tensors.
            # This may require adding a pointer to the source op for each output tensor.

        # Return the outputs.
        return output_data
