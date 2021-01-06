from pathlib import Path
import numpy as np

from dist_ir.executor import SequentialExecutor
from dist_ir.importer import import_from_onnx, mlir_parser
from dist_ir.ir import cpprint


def test_import_from_onnx():
    onnx_model_path = Path(__file__).parent / "mnist_gemm_bw_running.onnx"

    import_from_onnx(onnx_model_path)


def test_parser():
    mlir_str = """
    func @dp_inf(%wA: !dist.tensor<4x6xf32, 0>, %x: !dist.tensor<8x4xf32, 0>)
    -> (!dist.tensor<8x6xf32, 0>)
    {
    %xs = "Scatter"(%x) {dim = 0, devices = [1, 2]}
        : (!dist.tensor<8x4xf32, 0>) -> tuple<!dist.tensor<4x4xf32, 1>, !dist.tensor<4x4xf32, 2>>
    %wAs = "Broadcast"(%wA) {devices = [1, 2]}
        : (!dist.tensor<4x6xf32, 0>) -> tuple<!dist.tensor<4x6xf32, 1>, !dist.tensor<4x6xf32, 2>>
    %ys = "dist.pmap"(%xs, %wAs)
        ({
        ^bb0(%xi: !dist.tensor<4x4xf32, d>, %wAi: !dist.tensor<4x6xf32, d>):
            %yi = "MatMul"(%xi, %wAi)
            : (!dist.tensor<4x4xf32, d>, !dist.tensor<4x6xf32, d>) -> !dist.tensor<4x6xf32, d>
            "dist.return"(%yi) : (!dist.tensor<4x6xf32, d>) -> ()
        })
        {device_var = "d", devices = [1, 2]}
        : (tuple<!dist.tensor<4x4xf32, 1>, !dist.tensor<4x4xf32, 2>>,
        tuple<!dist.tensor<4x6xf32, 1>, !dist.tensor<4x6xf32, 2>>)
        -> tuple<!dist.tensor<4x6xf32, 1>, !dist.tensor<4x6xf32, 2>>
    %y = "Gather"(%ys) {device = 0, dim = 0}
        : (tuple<!dist.tensor<4x6xf32, 1>, !dist.tensor<4x6xf32, 2>>) -> !dist.tensor<8x6xf32, 0>
    return %y: !dist.tensor<8x6xf32, 0>
    }
    """
    modules = mlir_parser.parse_mlir_str(mlir_str)
    assert len(modules) == 1
    module = modules[0]
    cpprint(module)

    ex = SequentialExecutor("numpy")
    _wA = np.ones((4, 6))
    _x = np.arange(8 * 4).reshape((8, 4))
    res = ex.compute(module, {"%arg1": _x, "%arg0": _wA})

    # TODO fix concat's implementation in numpy register for this:
    # assert np.array_equal(res["%var4"], np.matmul(_x, _wA))
