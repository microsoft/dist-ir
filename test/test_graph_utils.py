from pathlib import Path

import dist_ir


def test_import_from_onnx():
    onnx_model_path = Path(__file__).parent / "../../../test/mnist_gemm_bw_running.onnx"
    backend = None

    dist_ir.graph.import_from_onnx(onnx_model_path, backend)
