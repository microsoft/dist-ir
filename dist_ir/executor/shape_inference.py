from ..ir.type import Float
from ..ir.type import Tensor
from ..ir.value import Value
from . import utils

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
    output_type = Tensor(dtype=inputs[0].type.dtype, shape=output_shape)
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
    outputs[0].type = Tensor(dtype=inputs[0].type.dtype, shape=output_shape)


def _infer_shapes_for_matmul_grad(op, inputs, outputs):
    for i, output in enumerate(outputs):
        output.type = copy.deepcopy(inputs[i].type)


def _infer_shapes_for_loss(op, inputs, outputs):
    outputs[0].type = Tensor(Float(), (1,))


def _infer_shapes_for_loss_grad(op, inputs, outputs):
    outputs[0].type = Tensor(Float(), (1,))


def _infer_shapes_for_pmap(op, inputs, outputs):
    value_name_map = op.get_metadata("value_name_map")
    value_map = {}
    for input in inputs:
        value_map[input.name] = input
    for output in outputs:
        value_map[output.name] = output
    submodule = op.get_submodule(0)
    for device in value_name_map:
        per_device_submodule = copy.deepcopy(submodule)
        _infer_shapes(per_device_submodule, value_name_map, value_map, device)


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


def _infer_shapes(module, value_name_map=None, value_map=None, device=None):
    """Helper function for inferring shapes.

    Inputs:
      module: The module to infer shapes for.
      value_name_map: A map from value name to another map between device
                      and mapped value name. This is used to resolve values
                      in a Pmap submodule for data parallelism.
      value_map: A map from value name to value.
      device: The device the module is executing on. This is relevant for data
              parallel execution where each replica is given to a particular device.
    """

    for op_name, op in module.get_ops().items():
        inputs = op.get_in_edges()
        outputs = op.get_out_edges()

        # If within a Pmap context, the module inputs and output values might be mapped
        # to partitioned values. We need to resolve these mappings to ensure we infer
        # shapes for the correct values on each device.
        mapped_inputs = utils.map_values(inputs, value_name_map, value_map, device)
        mapped_outputs = utils.map_values(outputs, value_name_map, value_map, device)

        ShapeInferenceRegister[op.op_type](op, mapped_inputs, mapped_outputs)


def infer_shapes(module):
    """Infers shapes for the given module."""
    _infer_shapes(module)
