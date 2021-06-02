import onnx
import pytest

from dist_ir.backend.onnx import export_onnx
from dist_ir.executor.type_inference import infer_types
from dist_ir.ir import Device, FunctionMaker, Op, cpprint, Value
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
    assert len(device_to_onnx_fns) == 1 and d1 in device_to_onnx_fns
    onnx.checker.check_model(device_to_onnx_fns[d1])


def test_non_onnx_op():
    d1 = Device(1, "gpu")
    fn = FunctionMaker()
    x = fn.add_input_value("x", Tensor(Float(), (4, 4), d1))
    op = Op(op_type="Dummy", inputs=(x,), output_types=(Tensor(Float(), (4, 4), d1),))
    fn.ops.append(op)
    y = op.outputs[0]
    fn.set_outputs((y,))
    fn = fn.finalize()
    cpprint(fn)

    device_to_onnx_fns = export_onnx(fn)
    with pytest.raises(onnx.onnx_cpp2py_export.checker.ValidationError):
        onnx.checker.check_model(device_to_onnx_fns[d1])
