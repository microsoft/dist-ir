from pathlib import Path

from dist_ir import import_from_onnx
from dist_ir.ir import Module, Topology
from dist_ir.ir.type import Float, Tensor
from dist_ir.ir import cpprint


def test_cpprint():
    module = Module()
    topology = Topology()

    d = topology.add_device("gpu")

    a = module.add_input_value("a", Tensor(dtype=Float(), shape=(4, 4), device=d))
    b = module.add_input_value("b", Tensor(dtype=Float(), shape=(4, 4), device=d))
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b])
    y = module.add_op("MatMul", "MatMul1", inputs=[x, b])
    module.finalize()

    cpprint(module)


def test_import_from_onnx():
    onnx_model_path = Path(__file__).parent / "mnist_gemm_bw_running.onnx"
    module = import_from_onnx(onnx_model_path)
    cpprint(module)


if __name__ == "__main__":
    test_cpprint()
    test_import_from_onnx()
    import prettyprinter

    print(prettyprinter.get_default_config())
