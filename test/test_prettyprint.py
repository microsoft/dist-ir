from pathlib import Path

from dist_ir.importer import import_from_onnx
from dist_ir.ir import FunctionMaker, Topology
from dist_ir.ir.type import Float, Tensor
from dist_ir.ir import cpprint


def test_cpprint():
    function = FunctionMaker()
    topology = Topology()

    d = topology.add_device("gpu")

    a = function.add_input_value("a", Tensor(dtype=Float(), shape=(4, 4), device=d))
    b = function.add_input_value("b", Tensor(dtype=Float(), shape=(4, 4), device=d))
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b])
    y = function.add_op("MatMul", "MatMul1", inputs=[x, b])
    function.finalize()

    cpprint(function)


def test_import_from_onnx():
    onnx_model_path = Path(__file__).parent / "mnist_gemm_bw_running.onnx"
    function = import_from_onnx(onnx_model_path)
    cpprint(function)
