from __future__ import annotations

from .shape_inference import *


class OpRegisterEntry:
    def __init__(self, shape_inference_func, cost_inference_func):
        self._shape_inference_func = shape_inference_func
        self._cost_inference_func = cost_inference_func

    def infer_shapes(
        self, input_shapes: tuple[Value, ...]
    ) -> tuple[tuple[int, ...], ...]:
        return self._shape_inference_func(input_shapes)

    def infer_costs(self, input_shapes: tuple[Value, ...]) -> float:
        return self._cost_inference_func(input_shapes)


# TODO: Add cost inference functions
OpRegister = {
    "Add": OpRegisterEntry(infer_shapes_for_matmul, None),
    "MatMul": OpRegisterEntry(infer_shapes_for_matmul, None),
}
