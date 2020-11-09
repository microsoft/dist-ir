from ..ir.type import Float
from ..ir.type import Tensor
from ..ir.value import Value

from typing import Tuple


def infer_shapes_for_matmul(
    input_shapes: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Tuple[Tuple[int, int]]:
    # TODO: Handle input tensors with > 2 dimensions
    if input_shapes[0][1] != input_shapes[1][0]:
        raise ValueError(
            f"Incompatible MatMul input shapes {input_shapes[0]} and {inputs_shapes[1]}"
        )
    output_shape = (input_shapes[0][0], input_shapes[1][1])
    return (output_shape,)


def infer_shapes_for_add(
    inputs_shapes: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Tuple[Tuple[int, int]]:
    # TODO: Handle broadcasting
    if input_shapes[0] != input_shapes[1]:
        raise ValueError(
            f"Incompatible MatMul input shapes {input_shapes[0]} and {inputs_shapes[1]}"
        )
    output_shape = (input_shapes[0][0], input_shapes[0][1])
    return (output_shape,)
