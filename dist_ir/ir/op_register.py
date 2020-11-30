from .device import Device
from .type import Tensor, TupleType
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
                f"Op {op.name}: Expected {len(self._input_types)} inputs, "
                f"got {len(inputs)}"
            )
        for i, (input, input_type) in enumerate(zip(inputs, self._input_types)):
            if not isinstance(input.type, input_type):
                raise ValueError(
                    f"Op {op.name}: Expected input of type {input_type} for "
                    f"input {i}, got input of type {input.type}"
                )

        # Verify that the number of output names is correct if specified.
        if output_names is not None and len(output_names) != len(self._output_types):
            raise ValueError(
                f"Op {op.name}: Expected {len(output_names)} outputs, "
                f"got {len(self._output_types)}"
            )

        # Construct the output values and add them to the op's out edge list.
        for i, output_type in enumerate(self._output_types):
            if output_names is not None and output_names[i] != "":
                output_name = output_names[i]
            else:
                output_name = f"{op.name}/{i}"
            op.add_out_edge(Value(output_name, value_type=output_type()))


class AllreduceOpRegisterEntry(OpRegisterEntry):
    # TODO: Remove this and handle generic types in OpRegisterEntry
    def infer_types(self, op, output_names=None):
        inputs = op.get_in_edges()
        if len(inputs) != 1:
            raise ValueError(f"Op {op.name}: Expected 1 input, got {len(inputs)}")
        elif not isinstance(inputs[0].type, TupleType):
            raise ValueError(
                f"Op {op.name}: Expected input of type {self._input_types[0]}, "
                f"got input of type {inputs[0].type}"
            )
        if output_names is not None:
            output_name = output_names[0]
        else:
            output_name = f"{op.name}/{0}"
        output_value_type = copy.deepcopy(inputs[0].type)
        op.add_out_edge(Value(name=output_name, value_type=output_value_type))


class BroadcastScatterOpRegisterEntry(OpRegisterEntry):
    # TODO: Remove this and handle generic types in OpRegisterEntry
    def infer_types(self, op, output_names=None):
        inputs = op.get_in_edges()
        devices = op.get_attribute("devices")
        if output_names is not None and len(output_names) != 1:
            raise ValueError(
                f"Op {op.name}: Expected 1 output name but got {len(output_names)}"
            )
        output_types = []
        for i, device in enumerate(devices):
            output_type = copy.deepcopy(inputs[0].type)
            output_type.set_device(device)
            if op.op_type == "Scatter":
                split_dim = op.get_attribute("split_dim")
                if isinstance(output_type, Tensor):
                    output_type.shape = None
            output_types.append(output_type)
        output_value = Value(output_names[0], value_type=TupleType(output_types))
        op.add_out_edge(output_value)


class PmapOpRegisterEntry(OpRegisterEntry):
    def infer_types(self, op, output_names=None):
        devices = op.get_attribute("devices")
        submodule = op.get_submodule(0)
        submodule_inputs = submodule.get_inputs()
        submodule_outputs = submodule.get_outputs()
        # TODO: If we want a more robust solution for nested pmaps, move the
        # parameterization over device variable to the module code
        # TODO: Handle multiple device types?
        d = Device.get_new_device_variable(devices[0].device_type)
        for in_edge in submodule_inputs:
            in_edge.type.set_device(d)

        # TODO: Change the submodule input names to indicate they are
        # parameterized over the devices

        for i, out_edge in enumerate(submodule_outputs):
            output_types = []
            for device in devices:
                output_type = copy.deepcopy(out_edge.type)
                output_type.set_device(device)
                output_types.append(output_type)
            if output_names is None:
                output_name = f"{out_edge.name}is"
            else:
                output_name = output_names[i]
            output_value = Value(output_name, value_type=TupleType(output_types))
            op.add_out_edge(output_value)


OpRegister = {
    "Add": OpRegisterEntry(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Allreduce": AllreduceOpRegisterEntry(
        input_types=[TupleType[Tensor]], output_types=[TupleType[Tensor]]
    ),
    "Broadcast": BroadcastScatterOpRegisterEntry(
        input_types=[Tensor], output_types=[TupleType[Tensor]]
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
    "Scatter": BroadcastScatterOpRegisterEntry(
        input_types=[Tensor], output_types=[TupleType[Tensor]]
    ),
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
