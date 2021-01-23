import onnx

from ..ir import FunctionMaker, Value
from ..ir.type import Bool, Float, Int32, Int64, Tensor


def get_dist_ir_dtype_from_onnx_dtype(onnx_dtype):
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


def import_from_onnx(onnx_model):
    # TODO: Remove prints?
    # TODO: Support types beyond Tensor
    onnx_model = onnx.load(onnx_model)
    dist_ir_function = FunctionMaker("foo")  # TODO get name?

    inputs = {}
    output_src = {}

    def add_input(value):
        if value.name in inputs:
            print(f"Skipping adding {value.name}; already an input value")
            return
        assert "ValueInfoProto" in str(type(value))
        assert hasattr(value, "type")
        assert hasattr(value.type, "tensor_type")
        dtype = get_dist_ir_dtype_from_onnx_dtype(value.type.tensor_type.elem_type)
        typ = Tensor(dtype=dtype)
        v = dist_ir_function.add_input_value(value.name, typ)
        inputs[value.name] = v

    def add_tensor(value):
        if value.name in inputs:
            print(f"Skipping adding {value.name}; already an input value")
            return
        assert "TensorProto" in str(type(value))
        assert hasattr(value, "dims")
        assert hasattr(value, "data_type")
        dtype = get_dist_ir_dtype_from_onnx_dtype(value.data_type)
        typ = Tensor(dtype=dtype, shape=tuple(value.dims))
        v = dist_ir_function.add_input_value(value.name, typ)
        inputs[value.name] = v

    """
    for value in onnx_model.graph.value_info:
        print(f"Adding input {value.name} from graph.value_info")
        add_input(value)
    print()
    """

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
        outputs = dist_ir_function.add_op(
            op_type=node.op_type,
            name=node.name,
            inputs=per_node_inputs,
            output_names=output_names,
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

    return dist_ir_function.finalize()
