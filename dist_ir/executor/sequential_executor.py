from .backend_register import BackendRegister
from ..ir.type import Tensor


class SequentialExecutor:
    def __init__(self, backend):
        if backend not in BackendRegister:
            raise ValueError(f"Unknown backend {backend}")
        self._backend = backend

    def _compute_op(self, op, inputs):
        """Executes the given op and returns its outputs."""
        op_type = op.op_type
        if op_type not in BackendRegister[self._backend]:
            raise NotImplementedError(
                f"No {self._backend} implementation found for op {op_type}"
            )
        impl = BackendRegister[self._backend][op_type]
        out_edges = op.get_out_edges()
        # TODO: Support multiple output values
        output_data = impl(*inputs)
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
                    input_value = input_data[input_name]
                elif input_name in output_data:
                    input_value = output_data[input_name]
                    consumers[input_name] -= 1
                else:
                    raise ValueError(f"Invalid input {input_name} for op {op_name}")
                inputs.append(input_value)

            # TODO: Support more than 1 out edge per op
            out_edges = op.get_out_edges()
            res = self._compute_op(op, inputs)
            output_data[out_edges[0].name] = res
            consumers[out_edges[0].name] = 1

            # Garbage collect the fully consumed output tensors.
            to_free = []
            for output_name in output_data:
                if consumers[output_name] == 0 and not module.is_output(output_name):
                    to_free.append(output_name)
            for output_name in to_free:
                del output_data[output_name]

        # Return the outputs.
        return output_data
