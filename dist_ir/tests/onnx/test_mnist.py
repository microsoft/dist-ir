import dist_ir

onnx_model_path = '/home/keshav/Downloads/mnist_gemm_bw.onnx'
backend = None

dist_ir.graph.import_from_onnx(onnx_model_path, backend)
