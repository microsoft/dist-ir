from __future__ import annotations

from .type import Float
from .type import Tensor
from .value import Value

import copy


def infer_shapes_for_matmul(
    input_shapes: tuple[tuple[int, int], tuple[int, int]]
) -> tuple[tuple[int, int]]:
    # TODO: Handle input tensors with > 2 dimensions
    if input_shapes[0][1] != input_shapes[1][0]:
        raise ValueError(
            f"Incompatible MatMul input shapes {input_shapes[0]} and {inputs_shapes[1]}"
        )
    output_shape = (input_shapes[0][0], input_shapes[1][1])
    return (output_shape,)


def infer_shapes_for_add(
    inputs_shapes: tuple[tuple[int, int], tuple[int, int]]
) -> tuple[tuple[int, int]]:
    # TODO: Handle broadcasting
    if input_shapes[0] != input_shapes[1]:
        raise ValueError(
            f"Incompatible MatMul input shapes {input_shapes[0]} and {inputs_shapes[1]}"
        )
    output_shape = copy.deepcopy(input_shapes[0])
    return (output_shape,)
