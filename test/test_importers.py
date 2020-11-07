from pathlib import Path

import dist_ir


def test_import_from_onnx():
    onnx_model_path = Path(__file__).parent / "mnist_gemm_bw_running.onnx"

    dist_ir.import_from_onnx(onnx_model_path)
