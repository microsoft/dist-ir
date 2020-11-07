from .op import Op
from .type import Float
from .type import Tensor
from .value import Value


def infer_shapes_for_op(op: Op):
    op.reset_out_edges()
    if op.op_type == "Add":
        infer_shapes_for_add(op)
    elif op.op_type == "Gemm":
        raise NotImplementedError(f"Op type Gemm has not been implemented")
    elif op.op_type == "MatMul":
        infer_shapes_for_matmul(op)
    else:
        raise ValueError(f"Invalid op type {op.op_type}")


def infer_shapes_for_matmul(op: Op):
    # TODO: Handle input tensors with > 2 dimensions
    in_edges = op.get_in_edges()
    if len(in_edges) != 2:
        raise ValueError(f"MatMul requires 2 inputs, op {op.name} has {len(in_edges)}")
    if in_edges[0].type.shape[1] != in_edges[1].type.shape[0]:
        raise ValueError(
            f"Incompatible MatMul input shapes {op.inputs[0].shape[1]} and {op.inputs[1].shape[0]}"
        )
    output_name = f"{op.name}/0"
    output_shape = (in_edges[0].type.shape[0], in_edges[1].type.shape[1])
    op.add_out_edge(Value(name=output_name, type=Tensor(Float(), shape=output_shape)))


def infer_shapes_for_add(op: Op):
    # TODO: Handle broadcasting
    in_edges = op.get_in_edges()
    if len(in_edges) != 2:
        raise ValueError(f"Add requires 2 inputs, op {op.name} has {len(in_edges)}")
    if in_edges[0].type.shape != in_edges[1].type.shape:
        raise ValueError(
            f"Incompatible Add input shapes {op.inputs[0].shape[1]} and {op.inputs[1].shape[0]}"
        )
    output_name = f"{op.name}/0"
    output_shape = (in_edges[0].type.shape[0], in_edges[0].type.shape[1])
    op.add_out_edge(Value(name=output_name, type=Tensor(Float(), shape=output_shape)))
