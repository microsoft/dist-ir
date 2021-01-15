import numpy as np

from dist_ir.ir import Device, FunctionMaker
from dist_ir.ir.type import Int, Float, Tensor
from dist_ir.transforms import ParallelMapTransform
from dist_ir.executor import SequentialExecutor
from dist_ir.executor.type_inference import infer_types


def attention_numpy(n, h, d, Q, K, V, W_Q, W_K, W_V, W_O):
    q = np.matmul(Q, W_Q)
    k = np.matmul(K, W_K)
    v = np.matmul(V, W_V)

    q = np.transpose(np.reshape(q, (n, h, d)), (1, 0, 2))
    k = np.transpose(np.reshape(k, (n, h, d)), (1, 2, 0))
    v = np.transpose(np.reshape(v, (n, h, d)), (1, 0, 2))
    y1 = np.matmul(np.matmul(q, k), v)
    y1 = np.reshape(np.transpose(y1, (1, 0, 2)), (n, h * d))
    y2 = np.matmul(y1, W_O)
    return y2


def test_attention():
    n = 64
    d = 64
    h = 8
    d_model = d * h

    function = FunctionMaker()

    devices = [Device(i, "gpu") for i in range(9)]

    Q = function.add_input_value("Q", Tensor(Float(), (n, d_model), devices[0]))
    K = function.add_input_value("K", Tensor(Float(), (n, d_model), devices[0]))
    V = function.add_input_value("V", Tensor(Float(), (n, d_model), devices[0]))
    W_Q = function.add_input_value(
        "W_Q", Tensor(Float(), (d_model, d_model), devices[0])
    )
    W_K = function.add_input_value(
        "W_K", Tensor(Float(), (d_model, d_model), devices[0])
    )
    W_V = function.add_input_value(
        "W_V", Tensor(Float(), (d_model, d_model), devices[0])
    )
    W_O = function.add_input_value(
        "W_O", Tensor(Float(), (d_model, d_model), devices[0])
    )
    split_head_shape = function.add_input_value(
        "split_head_shape", Tensor(Int(), (3,), devices[0])
    )
    original_shape = function.add_input_value(
        "original_shape", Tensor(Int(), (2,), devices[0])
    )

    q = function.add_op("MatMul", inputs=[Q, W_Q], output_names=["q"])
    k = function.add_op("MatMul", inputs=[K, W_K], output_names=["k"])
    v = function.add_op("MatMul", inputs=[V, W_V], output_names=["v"])

    q = function.add_op("Reshape", inputs=[q, split_head_shape], output_names=["q"])
    k = function.add_op("Reshape", inputs=[k, split_head_shape], output_names=["k"])
    v = function.add_op("Reshape", inputs=[v, split_head_shape], output_names=["v"])
    q = function.add_op(
        "Transpose", inputs=[q], attributes={"perm": (1, 0, 2)}, output_names=["q"]
    )
    k = function.add_op(
        "Transpose", inputs=[k], attributes={"perm": (1, 2, 0)}, output_names=["k"]
    )
    v = function.add_op(
        "Transpose", inputs=[v], attributes={"perm": (1, 0, 2)}, output_names=["v"]
    )

    a = function.add_op("MatMul", inputs=[q, k], output_names=["a"])
    y1 = function.add_op("MatMul", inputs=[a, v], output_names=["y1"])
    y1 = function.add_op(
        "Transpose", inputs=[y1], attributes={"perm": (1, 0, 2)}, output_names=["y1"]
    )
    y1 = function.add_op(
        "Reshape",
        inputs=[y1, original_shape],
        output_names=["y1"],
    )
    y2 = function.add_op("MatMul", inputs=[y1, W_O], output_names=["y2"])
    function = function.finalize()
    # function = infer_types(function, function.inputs)

    ex = SequentialExecutor("numpy")
    _Q = np.random.normal(size=(n, d_model))
    _K = np.random.normal(size=(n, d_model))
    _V = np.random.normal(size=(n, d_model))
    _W_Q = np.random.normal(size=(d_model, d_model))
    _W_K = np.random.normal(size=(d_model, d_model))
    _W_V = np.random.normal(size=(d_model, d_model))
    _W_O = np.random.normal(size=(d_model, d_model))
    _split_head_shape = [n, h, d]
    _original_shape = [n, d_model]

    orig_res = ex.compute(
        function,
        {
            function.inputs[0]: _Q,
            function.inputs[1]: _K,
            function.inputs[2]: _V,
            function.inputs[3]: _W_Q,
            function.inputs[4]: _W_K,
            function.inputs[5]: _W_V,
            function.inputs[6]: _W_O,
            function.inputs[7]: _split_head_shape,
            function.inputs[8]: _original_shape,
        },
    )
    np.testing.assert_array_almost_equal(
        orig_res[function.outputs[0]],
        attention_numpy(n, h, d, _Q, _K, _V, _W_Q, _W_K, _W_V, _W_O),
    )
