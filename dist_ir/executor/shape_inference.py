from ..ir.type import Float
from ..ir.type import Tensor
from ..ir.value import Value

import copy


def _error_invalid_shapes(input_shapes):
    raise ValueError(
        f"Op {op.name} (op.type): Incompatible input shapes {input_shapes}"
    )


def _infer_shapes_for_add(op, inputs, input_shapes, outputs):
    # TODO: Handle input tensors with > 2 dimensions
    if input_shapes[0] != input_shapes[1]:
        _error_invalid_shapes()

    output_shape = (input_shapes[0][0], input_shapes[0][1])
    output_type = Tensor(dtype=inputs[0].type.dtype, shape=output_shape)
    outputs[0].type = output_type


def _infer_shapes_for_allreduce(op, inputs, input_shapes, outputs):
    pass


def _infer_shapes_for_broadcast(op, inputs, input_shapes, outputs):

    for output in outputs:
        output.type.shape = copy.deepcopy(inputs[0].type.shape)


def _infer_shapes_for_matmul(op, inputs, input_shapes, outputs):
    # TODO: Handle input tensors with > 2 dimensions
    if input_shapes[0][1] != input_shapes[1][0]:
        _error_invalid_shapes()

    output_shape = (input_shapes[0][0], input_shapes[1][1])
    output_type = Tensor(dtype=inputs[0].type.dtype, shape=output_shape)
    outputs[0].type = output_type


def _infer_shapes_for_matmul_grad(op, inputs, input_shapes, outputs):
    for i, output in enumerate(outputs):
        output.type.shape = copy.deepcopy(input_shapes[i])


def _infer_shapes_for_loss(op, inputs, input_shapes, outputs):
    pass


def _infer_shapes_for_loss_grad(op, inputs, input_shapes, outputs):
    outputs[0].shape = Tensor(Float(), (1,))


def _infer_shapes_for_pmap(op, inputs, input_shapes, outputs):
    value_name_map = op.get_metadata("value_name_map")
    print(value_name_map)
    value_map = {}
    for input in inputs:
        value_map[input.name] = input
    for output in outputs:
        value_map[output.name] = output
    submodule = op.get_submodule(0)
    for device in value_name_map:
        infer_shapes(submodule, value_name_map, value_map, device)
        for output in submodule.get_outputs():
            mapped_output_name = value_name_map[device][output.name]
            if isinstance(output.type, Tensor):
                value_map[mapped_output_name].type.shape = output.type.shape


def _infer_shapes_for_scatter(op, inputs, input_shapes, outputs):
    split_dim = op.get_attribute("split_dim")
    num_splits = op.get_attribute("num_splits")

    output_shape = list(input_shapes[0])
    if output_shape[split_dim] % num_splits != 0:
        _error_invalid_shapes()
    output_shape[split_dim] //= num_splits

    for output in outputs:
        output.type.shape = tuple(output_shape)


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


def infer_shapes(module, value_name_map=None, value_map=None, device=None):
    for op_name, op in module.get_ops().items():
        print(f"Inferring shapes for op {op_name} ({op.op_type})")
        inputs = op.get_in_edges()
        outputs = op.get_out_edges()
        input_shapes = []
        for input in inputs:
            if value_name_map is not None and input.name in value_name_map[device]:
                input_name = value_name_map[device][input.name]
                input = value_map[input_name]
            if isinstance(input.type, Tensor):
                input_shapes.append(input.type.shape)
            else:
                input_shapes.append(None)

        ShapeInferenceRegister[op.op_type](op, inputs, input_shapes, outputs)
