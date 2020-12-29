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
        f"Op {op.name} ({op.op_type}): Incompatible input shapes {input_shapes}"
    )


def _infer_shapes_for_add(op, inputs, outputs):
    # TODO: Handle input tensors with > 2 dimensions
    input_shapes = _get_shapes(inputs)
    if input_shapes[0] != input_shapes[1]:
        _error_invalid_shapes(op, input_shapes)

    output_shape = input_shapes[0]
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
            output_type.shape = input_type.shape
        output_type.set_device(device)


def _infer_shapes_for_concat(op, inputs, outputs):
    input_shapes = _get_shapes(inputs)
    dim = op.get_attribute("dim")
    for i, (dim0, dim1) in enumerate(zip(input_shapes[0], input_shapes[1])):
        if i != dim and dim0 != dim1:
            _error_invalid_shapes(op, input_shapes)
    output_shape = list(input_shapes[0])
    output_shape[dim] += input_shapes[1][dim]
    outputs[0].type.shape = output_shape


def _infer_shapes_for_gather(op, inputs, outputs):
    dim = op.get_attribute("dim")
    device = op.get_attribute("device")
    output_shape = list(inputs[0].type.types[0].shape)
    for typ in inputs[0].type.types[1:]:
        output_shape[dim] += typ.shape[dim]
    outputs[0].type.dtype = inputs[0].type.types[0].dtype
    outputs[0].type.shape = output_shape
    outputs[0].type.set_device(device)


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
    input_shapes = _get_shapes(inputs)
    if input_shapes[0] != input_shapes[1]:
        _error_invalid_shapes(op, input_shapes)

    outputs[0].type = copy.deepcopy(inputs[0].type)


def _infer_shapes_for_loss_grad(op, inputs, outputs):
    input_shapes = _get_shapes(inputs)
    if input_shapes[0] != input_shapes[1]:
        _error_invalid_shapes(op, input_shapes)

    outputs[0].type = copy.deepcopy(inputs[0].type)


def _infer_shapes_for_pmap(op, inputs, outputs):
    submodule = op.get_submodule(0)

    for (pmap_input, submodule_input) in zip(inputs, submodule.get_inputs()):
        assert isinstance(pmap_input.type, TupleType)
        # TODO check that all elements of the tuple have the same type and, if
        # they are tensors, that they have the same shape
        if isinstance(submodule_input.type, Tensor):
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
    split_dim = op.get_attribute("dim")
    devices = op.get_attribute("devices")
    for (output_type, device) in zip(outputs[0].type.types, devices):
        if isinstance(output_type, Tensor) and isinstance(input_type, Tensor):
            output_shape = list(input_type.shape)
            output_shape[split_dim] //= len(devices)
            output_type.shape = tuple(output_shape)
        output_type.set_device(device)


def _infer_shapes_for_select(op, inputs, outputs):
    dim = op.get_attribute("dim")
    outputs[0].type.shape = inputs[0].type.types[dim].shape
    outputs[0].type.dtype = inputs[0].type.types[dim].dtype


def _infer_shapes_for_send(op, inputs, outputs):
    outputs[0].type.shape = inputs[0].type.shape
    outputs[0].type.dtype = inputs[0].type.dtype


def _infer_shapes_for_split(op, inputs, outputs):
    num_splits = op.get_attribute("num_splits")
    split_dim = op.get_attribute("dim")
    output_shape = list(inputs[0].type.shape)
    output_shape[split_dim] //= num_splits
    for typ in outputs[0].type.types:
        typ.shape = tuple(output_shape)
        typ.dtype = inputs[0].type.dtype


ShapeInferenceRegister = {
    "Add": _infer_shapes_for_add,
    "Allreduce": _infer_shapes_for_allreduce,
    "Broadcast": _infer_shapes_for_broadcast,
    "Concat": _infer_shapes_for_concat,
    "Gather": _infer_shapes_for_gather,
    "Loss": _infer_shapes_for_loss,
    "LossGrad": _infer_shapes_for_loss_grad,
    "MatMul": _infer_shapes_for_matmul,
    "MatMulGrad": _infer_shapes_for_matmul_grad,
    "Pmap": _infer_shapes_for_pmap,
    "Scatter": _infer_shapes_for_scatter,
    "Select": _infer_shapes_for_select,
    "Send": _infer_shapes_for_send,
    "Split": _infer_shapes_for_split,
}


def _infer_shapes(module):
    """Helper function for inferring shapes.

    Inputs:
      module: The module to infer shapes for.
    """

    for op_name, op in module.get_ops().items():
        inputs = op.get_in_edges()
        outputs = op.get_out_edges()

        # Invariant: types and shapes of input are already inferred
        for input in inputs:
            assert input.type is not None
            if isinstance(input.type, Tensor):
                if input.type.shape is None:
                    raise ValueError(f"Input {input.name} of op {op_name} has no shape")

        ShapeInferenceRegister[op.op_type](op, inputs, outputs)
        # TODO maybe the register gives back the output types and we can check
        # here if they match existing types (if any) and if not, replace them


def infer_shapes(module):
    """Infers shapes for the given module."""
    _infer_shapes(module)
