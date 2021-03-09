from functools import reduce
from operator import add, mul
import numpy as np
import onnx

from ..ir import FunctionMaker, Value
from ..ir.type import Bool, Float, Int32, Int64, Tensor


def _get_dist_ir_dtype_from_onnx_dtype(onnx_dtype):
    if onnx_dtype == 0:
        raise ValueError("Undefined onnx_dtype")
    elif onnx_dtype == 1:
        return Float()
    elif onnx_dtype == 6:
        return Int32()
    elif onnx_dtype == 7:
        return Int64()
    elif onnx_dtype == 9:
        return Bool()
    else:
        raise NotImplementedError(f"onnx_dtype {onnx_dtype}")


def _get_numpy_dtype_from_onnx_dtype(onnx_dtype):
    if onnx_dtype == 0:
        raise ValueError("Undefined onnx_dtype")
    elif onnx_dtype == 1:
        return np.float32
    elif onnx_dtype == 6:
        return np.int32
    elif onnx_dtype == 7:
        return np.int64
    elif onnx_dtype == 9:
        return bool
    else:
        raise NotImplementedError(f"onnx_dtype {onnx_dtype}")


def _parse_attribute(attr):
    key = attr.name
    attr_type = attr.type
    value = None
    if attr_type == 0:
        raise ValueError("Undefined attribute type")
    elif attr_type == 1:
        assert isinstance(attr.f, float)
        value = attr.f
    elif attr_type == 2:
        assert isinstance(attr.i, int)
        value = attr.i
    elif attr_type == 3:
        value = str(attr.s)
    elif attr_type == 4:
        raise NotImplementedError("Tensor attribute")
    elif attr_type == 5:
        raise NotImplementedError("Graph attribute")
    elif attr_type == 11:
        raise NotImplementedError("Sparse tensor attribute")
    elif attr_type == 6:
        value = tuple(attr.floats)
        for v in value:
            assert isinstance(v, float)
    elif attr_type == 7:
        value = tuple(attr.ints)
        for v in value:
            assert isinstance(v, int)
    elif attr_type == 8:
        value = tuple(attr.strings)
        for v in value:
            assert isinstance(v, str)
    elif attr_type == 9:
        raise NotImplementedError("Tensors attribute")
    elif attr_type == 10:
        raise NotImplementedError("Graphs attribute")
    elif attr_type == 12:
        raise NotImplementedError("Sparse tensors attribute")
    assert value is not None
    return key, value


def _parse_tensor_proto(tensor_proto):
    numpy_dtype = _get_numpy_dtype_from_onnx_dtype(tensor_proto.data_type)
    if len(tensor_proto.float_data) > 0:
        assert numpy_dtype == np.float32
        data = np.array(tensor_proto.float_data, dtype=numpy_dtype)
    elif len(tensor_proto.int32_data) > 0:
        assert numpy_dtype == np.int32
        data = np.array(tensor_proto.int32_data, dtype=numpy_dtype)
    elif len(tensor_proto.int64_data) > 0:
        assert numpy_dtype == np.int64
        data = np.array(tensor_proto.int64_data, dtype=numpy_dtype)
    else:
        assert len(tensor_proto.raw_data) > 0
        data = np.frombuffer(tensor_proto.raw_data, dtype=numpy_dtype)
    if len(tensor_proto.dims) > 0:
        assert reduce(mul, tensor_proto.dims) == len(data)
    else:
        assert len(data) == 1
    data = np.reshape(data, tensor_proto.dims)
    return data


def parse_tensor_from_file(path):
    tensor_proto = onnx.TensorProto()
    with open(path, "rb") as f:
        tensor_proto.ParseFromString(f.read())
    return _parse_tensor_proto(tensor_proto)


def import_from_onnx(onnx_model, default_device=None, parse_input_data=True):
    # TODO: Remove prints?
    # TODO: Support types beyond Tensor
    onnx_model = onnx.load(onnx_model)
    dist_ir_function = FunctionMaker("foo")  # TODO get name?

    inputs = {}
    input_data = {}
    output_src = {}

    def add_input(value):
        if value.name in inputs:
            print(f"Skipping adding {value.name}; already an input value")
            return
        assert "ValueInfoProto" in str(type(value))
        assert hasattr(value, "type")
        assert hasattr(value.type, "tensor_type")
        dtype = _get_dist_ir_dtype_from_onnx_dtype(value.type.tensor_type.elem_type)
        typ = Tensor(dtype=dtype, device=default_device)
        v = dist_ir_function.add_input_value(value.name, typ)
        inputs[value.name] = v

    def add_tensor(value):
        if value.name in inputs:
            print(f"Skipping adding {value.name}; already an input value")
            return
        assert "TensorProto" in str(type(value))
        dist_ir_dtype = _get_dist_ir_dtype_from_onnx_dtype(value.data_type)
        typ = Tensor(
            dtype=dist_ir_dtype, shape=tuple(value.dims), device=default_device
        )
        v = dist_ir_function.add_input_value(value.name, typ)
        inputs[value.name] = v
        if parse_input_data:
            input_data[v] = _parse_tensor_proto(value)

    for value in onnx_model.graph.input:
        print(f"Adding input {value.name} from graph.input")
        add_input(value)
    print()

    for value in onnx_model.graph.initializer:
        print(f"Adding input {value.name} from graph.initializer")
        add_tensor(value)
    print()

    for node in onnx_model.graph.node:
        per_node_inputs = []
        print(f"Getting inputs for node {node.name} ({node.op_type})...")
        for value in node.input:
            if value == "":
                assert "Optimizer" in node.name
                continue
            if value in inputs:
                print(f"Found input {value} in inputs")
                per_node_inputs.append(inputs[value])
            elif value in output_src:
                print(f"Found input {value} in output_src")
                per_node_inputs.append(output_src[value])
            else:
                raise ValueError(f"Could not find input {value}!")
        output_names = [v for v in node.output if v != ""]
        attributes = {k: v for k, v in [_parse_attribute(a) for a in node.attribute]}
        outputs = dist_ir_function.add_op(
            op_type=node.op_type,
            name=node.name,
            inputs=per_node_inputs,
            output_names=output_names,
            attributes=attributes,
        )
        # Match node's outputs with the output Values created in op:
        if len(node.output) == 1:
            assert isinstance(outputs, Value)
            outputs = [outputs]
        else:
            assert len(outputs) == len(output_names)
        for out_name, value in zip(output_names, outputs):
            if out_name == "":
                assert "Optimizer" in node.name
                continue
            assert out_name == value.name
            assert out_name not in output_src
            output_src[out_name] = value
            print(f"Found output {out_name}")
        print()

    return dist_ir_function.finalize(), input_data
