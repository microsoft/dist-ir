from .type import Tensor
from .value import Value

import copy


class OpRegisterEntry:
    def __init__(self, input_types, output_types):
        self._input_types = input_types
        self._output_types = output_types

    def infer_types(self, op, output_names=None):
        # Verify that number of inputs and input types match the expected input types.
        inputs = op.get_in_edges()
        if len(inputs) != len(self._input_types):
            raise ValueError(
                f"Op {op.name}: Expected {len(self._input_types)} inputs, got {len(inputs)}"
            )
        for i, (input, input_type) in enumerate(zip(inputs, self._input_types)):
            if not isinstance(input.type, input_type):
                raise ValueError(
                    f"Op {op.name}: Expected input of type {input_type} for input {i}, got input of type {input.type}"
                )

        # Verify that the number of output names is correct if specified.
        if output_names is not None and len(output_names) != len(self._output_types):
            raise ValueError(
                f"Op {op.name}: Expected {len(output_names)} outputs, got {len(self._output_types)}"
            )

        # Construct the output values and add them to the op's out edge list.
        for i, output_type in enumerate(self._output_types):
            if output_names is not None and output_names[i] != "":
                output_name = output_names[i]
            else:
                output_name = f"{op.name}/{i}"
            op.add_out_edge(
                Value(output_name, value_type=output_type(), device=op.device)
            )


class AllreduceOpRegisterEntry(OpRegisterEntry):
    def infer_types(self, op, output_names=None):
        if output_names is not None:
            output_name = output_names[0]
        else:
            output_name = f"{op.name}/{0}"
        op.add_out_edge(Value(name=output_name, value_type=None, device=op.device))


class BroadcastScatterOpRegisterEntry(OpRegisterEntry):
    def infer_types(self, op, output_names=None):
        inputs = op.get_in_edges()
        devices = op.get_attribute("devices")
        if output_names is not None and len(output_names) != len(devices):
            raise ValueError(
                f"Op {op.name}: Expected {len(devices)} output names but got "
                f"{len(output_names)}"
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
            output_value = Value(output_name, value_type=output_type, device=device)
            op.add_out_edge(output_value)


class PmapOpRegisterEntry(OpRegisterEntry):
    def infer_types(self, op, output_names=None):
        devices = op.get_attribute("devices")
        num_outputs = 0
        for device in devices:
            submodule = op.get_submodule(0)
            submodule_outputs = submodule.get_outputs()
            for out_edge in submodule_outputs:
                if output_names is not None:
                    output_name = output_names[num_outputs]
                else:
                    output_name = f"{out_edge.name}_{device}"
                output_value = Value(
                    output_name, value_type=copy.deepcopy(out_edge.type), device=device
                )
                op.add_out_edge(output_value)
                num_outputs += 1


OpRegister = {
    "Add": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Allreduce": AllreduceOpRegisterEntry(input_types=None, output_types=[Tensor]),
    "Broadcast": BroadcastScatterOpRegisterEntry(
        input_types=[Tensor], output_types=None
    ),
    "BroadcastGradientArgs": OpRegisterEntry(
        input_types=[Tensor, Tensor], output_types=[Tensor, Tensor]
    ),
    "Gemm": OpRegisterEntry(
        input_types=[Tensor, Tensor, Tensor], output_types=[Tensor]
    ),
    "Loss": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "LossGrad": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "MatMul": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "MatMulGrad": OpRegisterEntry(
        input_types=[Tensor, Tensor, Tensor], output_types=[Tensor, Tensor]
    ),
    "ReduceSumTraining": OpRegisterEntry(
        input_types=[Tensor, Tensor], output_types=[Tensor]
    ),
    "Relu": OpRegisterEntry(input_types=[Tensor], output_types=[Tensor]),
    "ReluGrad": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Reshape": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Opt": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Pmap": PmapOpRegisterEntry(input_types=None, output_types=None),
    "Scatter": BroadcastScatterOpRegisterEntry(input_types=[Tensor], output_types=None),
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
