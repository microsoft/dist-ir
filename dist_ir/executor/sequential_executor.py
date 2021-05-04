import numpy as np
from typing import Any, Dict, List, Sequence

from .absint import AbstractInterpreter, convert_impls_to_semantics
from .type_inference import _type_function
from .backend_register import BackendRegister
from ..ir import Device, Function, Op, Value
from ..ir.type import Int32, Int64, Float32, Float64, Tensor


class SequentialExecutor:
    def __init__(self, backend):
        if backend not in BackendRegister:
            raise ValueError(f"Unknown backend {backend}")
        semantics = convert_impls_to_semantics(BackendRegister[backend])
        self.interpreter = AbstractInterpreter(semantics=semantics)

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
                outs = self.compute(op.subfunctions[0], inps)
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

    def infer_types(self, function: Function, inputs: Sequence[Any]) -> Function:
        """Given a function and a list of input values, returns a new function where
        all values are typed.

        inputs: a list/tuple of Values, of the same length as function.inputs, but
        the names are irrelevant.
        """

        def _numpy_dtype_to_dist_ir_dtype(dtype):
            if dtype == np.int32:
                return Int32()
            elif dtype == np.int64:
                return Int64()
            elif dtype == np.float32:
                return Float32()
            elif dtype == np.float64:
                return Float64()
            else:
                raise NotImplementedError(f"Unrecognized NumPy dtype {dtype}")

        # Run reference execution to get the output shapes.
        state = self.interpreter.interpret(function, inputs)

        # Propagate devices seperately from shapes.
        device_map = {}
        for inp in function.inputs:
            device = inp.type.device
            device_map[inp] = device
        for op in function.ops:
            input_devices = [device_map[inp] for inp in op.inputs]
            if op.op_type == "MPIBroadcast" or op.op_type == "MPIScatter":
                output_devices = op.attributes["devices"]
            elif (
                op.op_type == "MPIGather"
                or op.op_type == "MPIReduce"
                or op.op_type == "Send"
            ):
                output_devices = [op.attributes["device"]]
            elif op.op_type == "MPIAllreduce" or op.op_type == "MPIAllgather":
                output_devices = input_devices
            else:
                input_device_set = set(d for d in input_devices if d is not None)
                if len(input_device_set) > 1:
                    raise ValueError(
                        "Op {op} has inputs from devices {set(input_devices)}!"
                    )
                elif len(input_device_set) == 1:
                    output_devices = [input_devices[0] for _ in range(len(op.outputs))]
                else:
                    output_devices = [None]
            for output, device in zip(op.outputs, output_devices):
                device_map[output] = device

        # Construct a map from value to type using the reference execution state.
        type_map = {}
        for key, value in state.env.items():
            if isinstance(value, np.int64):
                type_map[key] = Int64()
            elif isinstance(value, np.float32):
                type_map[key] = Float32()
            elif isinstance(value, np.float64):
                type_map[key] = Float64()
            elif isinstance(value, np.ndarray):
                dtype = _numpy_dtype_to_dist_ir_dtype(value.dtype)
                type_map[key] = Tensor(
                    shape=value.shape, dtype=dtype, device=device_map[key]
                )
            elif isinstance(value, tuple):
                dtype = _numpy_dtype_to_dist_ir_dtype(value[0].dtype)
                type_map[key] = tuple(
                    Tensor(shape=value[0].shape, dtype=dtype, device=device_map[key][i])
                    for i in range(len(value))
                )
            else:
                raise ValueError(f"Found value {value} of type {type(value)}!")

        # Return a new function with the correct types.
        return _type_function(function, type_map)
