import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

from ..executor.rank_projector import project
from ..ir.type import Bool, Int32, Int64, Float


def _get_onnx_dtype(dtype):
    if isinstance(dtype, type(Int32())):
        return AttributeProto.INT
    elif isinstance(dtype, type(Int64())):
        # TODO: Check this
        return AttributeProto.INT
    elif isinstance(dtype, type(Float())):
        # TODO: Split DistIR Float to Float16 and Float32
        return AttributeProto.FLOAT
    elif isinstance(dtype, type(Bool())):
        # TODO: Check this
        return AttributeProto.INT
    else:
        return AttributeProto.UNDEFINED


def _export_onnx_helper(fn):
    value_map = {}
    nodes = []
    for inp in fn.inputs:
        print(inp)
        assert inp.name is not None and inp.name not in value_map
        assert inp.type is not None
        assert inp.type.dtype is not None
        assert inp.type.shape is not None
        value_map[inp.name] = helper.make_tensor_value_info(
            name=inp.name,
            elem_type=_get_onnx_dtype(inp.type.dtype),
            shape=inp.type.shape,
        )
    for op in fn.ops:
        inputs = [value_map[inp.name].name for inp in op.inputs]
        for output in op.outputs:
            assert output.name is not None and output.name not in value_map
            assert output.type is not None
            assert output.type.dtype is not None
            assert output.type.shape is not None
            value_map[output.name] = helper.make_tensor_value_info(
                name=output.name,
                elem_type=_get_onnx_dtype(output.type.dtype),
                shape=output.type.shape,
            )
        outputs = [value_map[output.name].name for output in op.outputs]
        node = helper.make_node(
            op_type=op.op_type,
            inputs=inputs,
            outputs=outputs,
            name=op.name,
            **op.attributes,
        )
        nodes.append(node)
    graph_def = helper.make_graph(
        nodes=nodes,
        name=fn.name,
        inputs=[value_map[inp.name] for inp in fn.inputs],
        outputs=[value_map[output.name] for output in fn.outputs],
    )
    model_def = helper.make_model(graph_def)
    onnx.checker.check_model(model_def)
    return model_def


def export_onnx(fn):
    device_to_fns, groups = project(fn, tuple(v.type for v in fn.inputs))
    devices = sorted(device_to_fns.keys())
    device_to_onnx_fns = {}
    for device in sorted(devices):
        projected_fn = device_to_fns[device]
        onnx_fn = _export_onnx_helper(projected_fn)
        device_to_onnx_fns[device] = onnx_fn
    return device_to_onnx_fns
