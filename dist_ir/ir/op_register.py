from dataclasses import dataclass

from .device import Device
from .type import Tensor, TupleType
from .value import Value

import copy  # TODO shouldn't need to deepcopy anything in this file


@dataclass(frozen=True)
class OpRegisterEntry:
    num_inputs: int
    num_outputs: int


OpRegister = {
    "Add": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Allreduce": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Broadcast": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "BroadcastGradientArgs": OpRegisterEntry(num_inputs=2, num_outputs=2),
    "Concat": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Gather": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Gemm": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "Loss": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "LossGrad": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "MatMul": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "MatMulGrad": OpRegisterEntry(num_inputs=3, num_outputs=2),
    "ReduceSumTraining": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Relu": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "ReluGrad": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Reshape": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Opt": OpRegisterEntry(num_inputs=2, num_outputs=1),
    "Scatter": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Select": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "Send": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "SGDOptimizer": OpRegisterEntry(num_inputs=3, num_outputs=2),
    "Shape": OpRegisterEntry(num_inputs=1, num_outputs=1),
    "SoftmaxCrossEntropy": OpRegisterEntry(num_inputs=2, num_outputs=2),
    "SoftmaxCrossEntropyGrad": OpRegisterEntry(num_inputs=3, num_outputs=1),
    "Split": OpRegisterEntry(num_inputs=1, num_outputs=1),
}


# TODO move the rest of this file to the new combined shape/type inference


class OpRegisterEntry_:
    def __init__(self, input_types, output_types):
        self._input_types = input_types
        self._output_types = output_types

    def infer_types(self, op, output_names=None):
        # Verify that number of inputs and input types match the expected input types.
        inputs = op.in_edges
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


class AllreduceOpRegisterEntry(OpRegisterEntry_):
    # TODO: Remove this and handle generic types in OpRegisterEntry_
    def infer_types(self, op, output_names=None):
        inputs = op.in_edges
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


class BroadcastScatterOpRegisterEntry(OpRegisterEntry_):
    # TODO: Remove this and handle generic types in OpRegisterEntry_
    def infer_types(self, op, output_names=None):
        inputs = op.in_edges
        devices = op.attributes["devices"]
        if output_names is not None and len(output_names) != 1:
            raise ValueError(
                f"Op {op.name}: Expected 1 output name but got {len(output_names)}"
            )
        output_types = []
        assert len(inputs) == 1 and isinstance(inputs[0].type, Tensor)
        in_type = inputs[0].type
        shape = in_type.shape if op.op_type == "Broadcast" else None
        for device in devices:
            output_type = Tensor(dtype=in_type.dtype, shape=shape, device=device)
            output_types.append(output_type)
        output_value = Value(output_names[0], value_type=TupleType(output_types))
        op.add_out_edge(output_value)


class PmapOpRegisterEntry(OpRegisterEntry_):
    def infer_types(self, op, output_names=None):
        # TODO I think we should figure out the devices used by looking at the
        # types of the inputs
        devices = op.attributes["devices"]
        subfunction = op.subfunctions[0]
        subfunction_inputs = subfunction.inputs
        subfunction_outputs = subfunction.outputs
        # TODO: If we want a more robust solution for nested pmaps, move the
        # parameterization over device variable to the function code
        # TODO: Handle multiple device types?
        # TODO the device should be created by parser, not here, because it is
        # needed to distinguish between inner and outer device vars in nested
        # pmaps
        d = Device.get_new_device_variable(devices[0].device_type)
        for in_edge in subfunction_inputs:
            in_edge.type.set_device(d)
        op._attributes["device_var"] = d

        # TODO: Change the subfunction input names to indicate they are
        # parameterized over the devices

        for i, out_edge in enumerate(subfunction_outputs):
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


class SelectOpRegisterEntry(OpRegisterEntry_):
    def infer_types(self, op, output_names=None):
        inputs = op.in_edges
        dim = op.attributes["dim"]
        output_value = Value(
            output_names[0], value_type=copy.deepcopy(inputs[0].type.types[dim])
        )
        op.add_out_edge(output_value)


class SendOpRegisterEntry(OpRegisterEntry_):
    def infer_types(self, op, output_names=None):
        inputs = op.in_edges
        assert len(inputs) == 1 and isinstance(inputs[0].type, Tensor)
        input_type = inputs[0].type
        device = op.attributes["device"]
        output_value_type = Tensor(input_type.dtype, input_type.shape, device)
        output_value = Value(output_names[0], value_type=output_value_type)
        op.add_out_edge(output_value)


class SplitOpRegisterEntry(OpRegisterEntry_):
    def infer_types(self, op, output_names=None):
        inputs = op.in_edges
        num_splits = op.attributes["num_splits"]
        output_types = []
        for i in range(num_splits):
            output_type = copy.deepcopy(inputs[0].type)
            output_type.shape = None
            output_types.append(output_type)
        output_value = Value(output_names[0], value_type=TupleType(output_types))
        op.add_out_edge(output_value)


OpRegister_ = {
    "Add": OpRegisterEntry_(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Allreduce": AllreduceOpRegisterEntry(
        input_types=[TupleType], output_types=[TupleType]
    ),
    "Broadcast": BroadcastScatterOpRegisterEntry(
        input_types=[Tensor], output_types=[TupleType]
    ),
    "BroadcastGradientArgs": OpRegisterEntry_(
        input_types=[Tensor, Tensor], output_types=[Tensor, Tensor]
    ),
    "Concat": OpRegisterEntry_(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Gather": OpRegisterEntry_(input_types=[TupleType], output_types=[Tensor]),
    "Gemm": OpRegisterEntry_(
        input_types=[Tensor, Tensor, Tensor], output_types=[Tensor]
    ),
    "Loss": OpRegisterEntry_(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "LossGrad": OpRegisterEntry_(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "MatMul": OpRegisterEntry_(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "MatMulGrad": OpRegisterEntry_(
        input_types=[Tensor, Tensor, Tensor], output_types=[Tensor, Tensor]
    ),
    "ReduceSumTraining": OpRegisterEntry_(
        input_types=[Tensor, Tensor], output_types=[Tensor]
    ),
    "Relu": OpRegisterEntry_(input_types=[Tensor], output_types=[Tensor]),
    "ReluGrad": OpRegisterEntry_(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Reshape": OpRegisterEntry_(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Opt": OpRegisterEntry_(input_types=[Tensor, Tensor], output_types=[Tensor]),
    "Pmap": PmapOpRegisterEntry(input_types=None, output_types=None),
    "Scatter": BroadcastScatterOpRegisterEntry(
        input_types=[Tensor], output_types=[TupleType]
    ),
    "Select": SelectOpRegisterEntry(input_types=[TupleType], output_types=[Tensor]),
    "Send": SendOpRegisterEntry(input_types=[Tensor], output_types=[Tensor]),
    "SGDOptimizer": OpRegisterEntry_(
        input_types=[Tensor, Tensor, Tensor], output_types=[Tensor, Tensor]
    ),
    "Shape": OpRegisterEntry_(input_types=[Tensor], output_types=[Tensor]),
    "SoftmaxCrossEntropy": OpRegisterEntry_(
        input_types=[Tensor, Tensor], output_types=[Tensor, Tensor]
    ),
    "SoftmaxCrossEntropyGrad": OpRegisterEntry_(
        input_types=[Tensor, Tensor, Tensor], output_types=[Tensor]
    ),
    "Split": SplitOpRegisterEntry(input_types=[Tensor], output_types=[TupleType]),
}
