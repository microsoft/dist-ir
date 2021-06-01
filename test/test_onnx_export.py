from dist_ir.backend.onnx import export_onnx
from dist_ir.executor.type_inference import infer_types
from dist_ir.ir import Device, FunctionMaker, cpprint, Value
from dist_ir.ir.type import Float, Tensor
from dist_ir.ir.topology import Topology

def test_single_device():
    d1 = Device(1, "gpu")
    fn = FunctionMaker()
    x = fn.add_input_value("x", Tensor(Float(), (4, 4), d1))
    y = fn.add_op("MatMul", inputs=(x, x))
    fn.set_outputs((y,))
    fn = fn.finalize()
    fn = infer_types(fn, fn.inputs) 
    cpprint(fn)

    device_to_onnx_fns = export_onnx(fn)

if __name__=='__main__':
    test_single_device()
