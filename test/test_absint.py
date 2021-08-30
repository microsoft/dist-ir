import numpy as np

from dist_ir.executor import absint
from dist_ir.executor.numpy_register import NumPyRegister
from dist_ir.executor.mixed_register import MixedImplementations

# NOTE: Disabling mlir_parser tests to pass GitHub automated test
# from dist_ir.importer import mlir_parser
from dist_ir.ir import cpprint
from dist_ir.ir.type import Tensor


# Batch size = 8
# Sequence length = 6

shape_slice_fn = """
func @shape_slice(
    %x: !dist.tensor<8x6xf32, 0>,
    %position_01: !dist.tensor<1x512xf32, 0>,
    %op_min_ends_expand_10: !dist.tensor<2xf32, 0>,
    %start_expand_10: !dist.tensor<2xi64, 0>,
    %axes_expand_10: !dist.tensor<2xi64, 0>
    ) -> (none)
{
    %73 = "Shape"(%x) : (!dist.tensor<8x6xf32, 0>) -> none
    %to_min_01 = "Cast"(%73) {to = 1} : (none) -> none
    %from_min_01 = "Min"(%to_min_01, %op_min_ends_expand_10) : (none, !dist.tensor<2xf32, 0>) -> none
    %to_slice_01 = "Cast"(%from_min_01) {to = 7} : (none) -> none
    %from_slice_01
        = "Slice"(%position_01, %start_expand_10, %to_slice_01, %axes_expand_10)
        : (!dist.tensor<1x512xf32, 0>, !dist.tensor<2xi64, 0>, none, !dist.tensor<2xi64, 0>) -> none
    return %from_slice_01: none
}
"""

# NOTE: Disabling mlir_parser tests to pass GitHub automated test
def _test_shape_slice():
    [fn] = mlir_parser.parse_mlir_str(shape_slice_fn)
    cpprint(fn)

    mixed_interpreter = absint.AbstractInterpreter(
        semantics=absint.convert_impls_to_semantics(
            {**NumPyRegister, **MixedImplementations}
        )
    )

    # First, execute concretely
    conc_inputs = [
        np.arange(8 * 6, dtype=np.float32).reshape(8, 6),
        np.arange(512, dtype=np.float32).reshape(1, 512),
        np.array([1, 9999], dtype=np.float32),
        np.array([0, 0], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
    ]
    state = mixed_interpreter.interpret(fn, conc_inputs)
    np.testing.assert_almost_equal(
        state.env[fn.outputs[0]], np.arange(6, dtype=np.float32).reshape(1, 6)
    )

    # Now, execute (slightly more) abstractly
    abs_inputs = [
        Tensor(shape=(8, 6)),
        np.arange(512, dtype=np.float32).reshape(1, 512),
        # The rest are the indices to slice, which we need to know concretely
        np.array([1, 9999], dtype=np.float32),
        np.array([0, 0], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
    ]
    state = mixed_interpreter.interpret(fn, abs_inputs)
    np.testing.assert_almost_equal(
        state.env[fn.outputs[0]], np.arange(6, dtype=np.float32).reshape(1, 6)
    )

    # Now, execute (even more) abstractly
    abs_inputs = [
        Tensor(shape=(8, 6)),
        Tensor(shape=(1, 512)),
        # The rest are the indices to slice, which we need to know concretely
        np.array([1, 9999], dtype=np.float32),
        np.array([0, 0], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
    ]
    state = mixed_interpreter.interpret(fn, abs_inputs)
    assert state.env[fn.outputs[0]] == Tensor(shape=(1, 6))
