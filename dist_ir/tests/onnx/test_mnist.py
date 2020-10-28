import dist_ir

# TODO: Fix this path (command line arg or os.path.dirname?)
onnx_model_path = "../../../test/mnist_gemm_bw_running.onnx"
backend = None

dist_ir.graph.import_from_onnx(onnx_model_path, backend)
