from ..ir.type import Float
from ..ir.type import Tensor, TupleType
from ..ir.value import Value
from ..ir.device import Device

import copy


def _get_shapes(values):
    shapes = []
    for value in values:
        if isinstance(value.type, Tensor):
            shapes.append(value.type.shape)
        else:
            shapes.append(None)
    return shapes


def _error_invalid_shapes(op, input_shapes):
    raise ValueError(
        f"Op {op.name} (op.type): Incompatible input shapes {input_shapes}"
    )


def _infer_shapes_for_add(op, inputs, outputs):
    # TODO: Handle input tensors with > 2 dimensions
    input_shapes = _get_shapes(inputs)
    if input_shapes[0] != input_shapes[1]:
        _error_invalid_shapes(op, input_shapes)

    output_shape = (input_shapes[0][0], input_shapes[0][1])
    output_type = Tensor(
        dtype=inputs[0].type.dtype, shape=output_shape, device=inputs[0].type.device
    )
    outputs[0].type = output_type


def _infer_shapes_for_allreduce(op, inputs, outputs):
    outputs[0].type = copy.deepcopy(inputs[0].type)


def _infer_shapes_for_broadcast(op, inputs, outputs):
    input_type = inputs[0].type
    devices = op.get_attribute("devices")
    output_types = []
    for (output_type, device) in zip(outputs[0].type.types, devices):
        if isinstance(output_type, Tensor) and isinstance(input_type, Tensor):
            output_type.shape = copy.deepcopy(input_type.shape)
        output_type.device = device


def _infer_shapes_for_matmul(op, inputs, outputs):
    # TODO: Handle input tensors with > 2 dimensions
    input_shapes = _get_shapes(inputs)
    if input_shapes[0][1] != input_shapes[1][0]:
        _error_invalid_shapes(op, input_shapes)
    output_shape = (input_shapes[0][0], input_shapes[1][1])
    outputs[0].type = Tensor(
        dtype=inputs[0].type.dtype, shape=output_shape, device=inputs[0].type.device
    )


def _infer_shapes_for_matmul_grad(op, inputs, outputs):
    for i, output in enumerate(outputs):
        output.type = copy.deepcopy(inputs[i].type)


def _infer_shapes_for_loss(op, inputs, outputs):
    outputs[0].type = Tensor(dtype=Float(), shape=(1,), device=inputs[0].type.device)


def _infer_shapes_for_loss_grad(op, inputs, outputs):
    outputs[0].type = Tensor(dtype=Float(), shape=(1,), device=inputs[0].type.device)


def _infer_shapes_for_pmap(op, inputs, outputs):
    submodule = op.get_submodule(0)

    for (pmap_input, submodule_input) in zip(inputs, submodule.get_inputs()):
        if isinstance(submodule_input.type, Tensor):
            assert isinstance(pmap_input.type, TupleType)
            submodule_input.type.shape = pmap_input.type.types[0].shape

    _infer_shapes(submodule)

    for (pmap_output, submodule_output) in zip(outputs, submodule.get_outputs()):
        if isinstance(submodule_output.type, Tensor):
            assert isinstance(pmap_output.type, TupleType)
            for pmap_output_type in pmap_output.type.types:
                pmap_output_type.shape = submodule_output.type.shape
                pmap_output_type.dtype = submodule_output.type.dtype


def _infer_shapes_for_scatter(op, inputs, outputs):
    input_type = inputs[0].type
    split_dim = op.get_attribute("split_dim")
    devices = op.get_attribute("devices")
    for (output_type, device) in zip(outputs[0].type.types, devices):
        if isinstance(output_type, Tensor) and isinstance(input_type, Tensor):
            output_shape = list(input_type.shape)
            output_shape[split_dim] //= len(devices)
            output_type.shape = tuple(output_shape)
        output_type.device = device


ShapeInferenceRegister = {
    "Add": _infer_shapes_for_add,
    "Allreduce": _infer_shapes_for_allreduce,
    "Broadcast": _infer_shapes_for_broadcast,
    "Loss": _infer_shapes_for_loss,
    "LossGrad": _infer_shapes_for_loss_grad,
    "MatMul": _infer_shapes_for_matmul,
    "MatMulGrad": _infer_shapes_for_matmul_grad,
    "Pmap": _infer_shapes_for_pmap,
    "Scatter": _infer_shapes_for_scatter,
}


def _infer_shapes(module):
    """Helper function for inferring shapes.

    Inputs:
      module: The module to infer shapes for.
    """

    for op_name, op in module.get_ops().items():
        inputs = op.get_in_edges()
        outputs = op.get_out_edges()

        ShapeInferenceRegister[op.op_type](op, inputs, outputs)


def infer_shapes(module):
    """Infers shapes for the given module."""
    _infer_shapes(module)
