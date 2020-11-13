from .type import Tensor
from .value import Value

import copy


class OpRegisterEntry:
    def __init__(self, input_types, output_types):
        self._input_types = input_types
        self._output_types = output_types

    def _infer_types_for_allreduce(self, op, output_names):
        if output_names is not None:
            output_name = output_names[0]
        else:
            output_name = f"{op.name}/{0}"
        op._out_edges.append(Value(name=output_name, value_type=None))

    def _infer_types_for_pmap(self, op, output_names):
        devices = op.get_attribute("devices")
        num_outputs = 0
        for device in devices:
            for out_edge in op._submodules[0].outputs:
                if output_names is not None:
                    output_name = output_names[num_outputs]
                else:
                    output_name = f"{out_edge.name}_{device}"
                output_value = Value(
                    output_name, value_type=copy.deepcopy(out_edge.type), device=device
                )
                op._out_edges.append(output_value)
                num_outputs += 1

    def _infer_types_for_broadcast_and_scatter(self, op, output_names):
        inputs = op.get_in_edges()
        devices = op.get_attribute("devices")
        if output_names is not None and len(output_names) != len(devices):
            raise ValueError(
                f"Op {op.name}: Expected {len(devices)} output names but got {len(output_names)}"
            )
        for i, device in enumerate(devices):
            if output_names is not None:
                output_name = output_names[i]
            else:
                output_name = f"{op_name}/{i}"
            output_type = copy.deepcopy(inputs[0].type)
            if op.op_type == "Scatter":
                split_dim = op.get_attribute("split_dim")
                if isinstance(output_type, Tensor):
                    output_type.shape = None
            output_value = Value(output_name, value_type=output_type, device=devices[i])
            op._out_edges.append(output_value)

    def infer_types(self, op, output_names=None):
        # Handle ops with a variable number of inputs.
        if self._input_types is None:
            if op.op_type == "Allreduce":
                self._infer_types_for_allreduce(op, output_names)
                return
            elif op.op_type == "Pmap":
                self._infer_types_for_pmap(op, output_names)
                return
            else:
                raise ValueError(f"Op {op.name}: No input types specified")

        # Verify that number of inputs and input types match the expected input types.
        inputs = op.get_in_edges()
        if len(inputs) != len(self._input_types):
            raise ValueError(
                f"Op {op.name}: Expected {len(self._input_types)} inputs, got {len(inputs)}"
            )
        for i, (input, input_type) in enumerate(zip(inputs, self._input_types)):
            if not isinstance(input.type, input_type):
                raise ValueError(
                    f"Op {op.name}: Expected input of type {input_type} for input {i}, got input of type {type(input)}"
                )

        # Handle ops with a variable number of outputs.
        if self._output_types is None:
            if op.op_type == "Broadcast" or op.op_type == "Scatter":
                self._infer_types_for_broadcast_and_scatter(op, output_names)
                return
            else:
                raise ValueError(f"Op {op.name}: Output types were not specified")

        # Verify that the number of output names is correct if specified.
        if output_names is not None and len(output_names) != len(self._output_types):
            raise ValueError(
                f"Op {op.name}: Expected {len(output_names)} outputs, got {len(self._output_types)}"
            )

        # Construct the output values and add them to the op's out edge list.
        devices = set([input.device for input in inputs])
        if len(devices) > 1:
            raise ValueError(
                f"Op {op.name}: Received inputs from multiple devices ({devices})"
            )
        device = list(devices)[0]
        for i, output_type in enumerate(self._output_types):
            if output_names is not None and output_names[i] != "":
                output_name = output_names[i]
            else:
                output_name = f"{op.name}/{i}"
            op._out_edges.append(
                Value(output_name, value_type=output_type(), device=device)
            )


OpRegister = {
    "Add": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Allreduce": OpRegisterEntry(
        input_types=None, output_types=[Tensor]
    ),  # TODO: Make input type a tuple of Tensors?
    "Broadcast": OpRegisterEntry(input_types=[Tensor], output_types=None),
    "BroadcastGradientArgs": OpRegisterEntry(
        input_types=[Tensor, Tensor], output_types=[Tensor, Tensor]
    ),
    "Gemm": OpRegisterEntry(
        input_types=[Tensor, Tensor, Tensor], output_types=[Tensor]
    ),
    "MatMul": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "ReduceSumTraining": OpRegisterEntry(
        input_types=[Tensor, Tensor], output_types=[Tensor]
    ),
    "Relu": OpRegisterEntry(input_types=[Tensor], output_types=[Tensor]),
    "ReluGrad": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Reshape": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Pmap": OpRegisterEntry(
        input_types=None, output_types=None
    ),  # TODO: Make input and output types tuples of Tensors?
    "Scatter": OpRegisterEntry(input_types=[Tensor], output_types=None),
    "SGDOptimizer": OpRegisterEntry(
        input_types=[Tensor, Tensor, Tensor], output_types=[Tensor, Tensor]
    ),
    "Shape": OpRegisterEntry(input_types=[Tensor], output_types=[Tensor]),
    "SoftmaxCrossEntropy": OpRegisterEntry(
        input_types=[Tensor, Tensor], output_types=[Tensor, Tensor]
    ),
    "SoftmaxCrossEntropyGrad": OpRegisterEntry(
        input_types=[Tensor, Tensor, Tensor], output_types=[Tensor]
    ),
}
