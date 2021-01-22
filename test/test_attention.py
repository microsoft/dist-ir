import numpy as np

from dist_ir.ir import Device, FunctionMaker
from dist_ir.ir.type import Int, Float, Tensor
from dist_ir.transforms import shard_transform
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


def megatron_attention_numpy(n, h, d, Q, K, V, W_Q, W_K, W_V, W_O):
    qs = [np.matmul(Q, W_Q[:, i * d : (i + 1) * d]) for i in range(h)]
    ks = [np.matmul(K, W_K[:, i * d : (i + 1) * d]) for i in range(h)]
    vs = [np.matmul(V, W_V[:, i * d : (i + 1) * d]) for i in range(h)]

    y1s = [np.matmul(np.matmul(qs[i], ks[i].T), vs[i]) for i in range(h)]
    y2s = [np.matmul(y1s[i], W_O[i * d : (i + 1) * d]) for i in range(h)]

    # Allreduce
    return np.sum(y2s, axis=0)


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


def test_megatron():
    n = 64
    h = 8
    d = 64
    d_model = h * d
    Q = np.random.normal(size=(n, d_model))
    K = np.random.normal(size=(n, d_model))
    V = np.random.normal(size=(n, d_model))
    W_Q = np.random.normal(size=(d_model, d_model))
    W_K = np.random.normal(size=(d_model, d_model))
    W_V = np.random.normal(size=(d_model, d_model))
    W_O = np.random.normal(size=(d_model, d_model))
    x = attention_numpy(n, h, d, Q, K, V, W_Q, W_K, W_V, W_O)
    y = megatron_attention_numpy(n, h, d, Q, K, V, W_Q, W_K, W_V, W_O)
    np.testing.assert_array_almost_equal(x, y)


def self_attention_before(n, w_1, w_2, x):  # (h, 3h), (h, h), (s, b, h)
    s, b, h = x.shape
    x1 = np.matmul(x, w_1)  # (s, b, 3h)

    x2 = np.reshape(x1, (s, b, n, 3 * h // n))
    q, k, v = np.split(x2, 3, axis=3)  # (s, b, n, h/n)

    q = np.reshape(q, (s, b * n, h // n))
    q = np.transpose(q, (1, 0, 2))  # (bn, s, h/n)
    k = np.reshape(k, (s, b * n, h // n))
    k = np.transpose(k, (1, 2, 0))  # (bn, h/n, s)
    v = np.reshape(v, (s, b * n, h // n))
    v = np.transpose(v, (1, 0, 2))  # (bn, s, h/n)

    y1 = np.matmul(q, k)  # (bn, s, s)

    # Ignoring a scale mask, softmax, and dropout here

    y2 = np.matmul(y1, v)  # (bn, s, h/n)
    # No idea why megatron goes through these layout changes:
    y2 = np.reshape(y2, (b, n, s, h // n))
    y2 = np.transpose(y2, (2, 0, 1, 3))  # (s, b, n, h/n)
    y2 = np.reshape(y2, (s, b, h))

    # RowParallelLinear
    z = np.matmul(y2, w_2)  # (s, b, h)

    return z


def self_attention_after(n, w_1, w_2, x, p=2):  # (h, 3h), (h, h), (s, b, h)
    s, b, h = x.shape

    w_1s = np.split(w_1, p, axis=1)
    w_2s = np.split(w_2, p, axis=0)
    xs = [x for i in range(p)]

    res = []
    for w_1, w_2, x in zip(w_1s, w_2s, xs):  # (h, 3h/p), (h/p, h), (s, b, h)
        x1 = np.matmul(x, w_1)  # (s, b, 3h/p)

        # TODO how do we get p for this reshape?
        x2 = np.reshape(x1, (s, b, n // p, 3 * h // n))
        q, k, v = np.split(x2, 3, axis=3)  # (s, b, n/p, h/n)

        q = np.reshape(q, (s, b * n // p, h // n))
        q = np.transpose(q, (1, 0, 2))  # (bn/p, s, h/n)
        k = np.reshape(k, (s, b * n // p, h // n))
        k = np.transpose(k, (1, 2, 0))  # (bn/p, h/n, s)
        v = np.reshape(v, (s, b * n // p, h // n))
        v = np.transpose(v, (1, 0, 2))  # (bn/p, s, h/n)

        y1 = np.matmul(q, k)  # (bn/p, s, s)

        # Ignoring a scale mask, softmax, and dropout here

        y2 = np.matmul(y1, v)  # (bn/p, s, h/n)
        # No idea why megatron goes through these layout changes:
        y2 = np.reshape(y2, (b, n // p, s, h // n))
        y2 = np.transpose(y2, (2, 0, 1, 3))  # (s, b, n/p, h/n)
        y2 = np.reshape(y2, (s, b, h // p))

        # RowParallelLinear
        z = np.matmul(y2, w_2)  # (s, b, h)
        res.append(z)

    # all-reduce
    z = sum(res)
    return z


def test_manual_transform_on_self_attention():
    s = 3  # sequence length
    b = 4  # batch size
    n = 2  # num attention heads
    h = 6  # hidden size
    p = 2  # num partitions

    x = np.ones((s, b, h))  # model input/output of previous layer
    w_1 = np.ones((h, 3 * h))
    w_2 = np.ones((h, h))

    z1 = self_attention_before(n, w_1, w_2, x)
    print(z1)
    z2 = self_attention_after(n, w_1, w_2, x)
    print(z2)

    assert np.allclose(z1, z2)


def test_dist_ir_self_attention():
    s = 3  # sequence length
    b = 4  # batch size
    n = 2  # num attention heads
    h = 6  # hidden size
    p = 2  # num partitions

    devices = [Device(i, "gpu") for i in range(p + 1)]

    # Original function
    function = FunctionMaker()
    x = function.add_input_value(
        "x", Tensor(dtype=Float(), shape=(s, b, h), device=devices[0])
    )
    w_1 = function.add_input_value(
        "w1", Tensor(dtype=Float(), shape=(h, 3 * h), device=devices[0])
    )
    w_2 = function.add_input_value(
        "w2", Tensor(dtype=Float(), shape=(h, h), device=devices[0])
    )
    x2_shape = function.add_input_value(
        "x2_shape", Tensor(dtype=Int(), shape=(4,), device=devices[0])
    )
    qkv_shape = function.add_input_value(
        "qkv_shape", Tensor(dtype=Int(), shape=(3,), device=devices[0])
    )
    y2_shape_before_transpose = function.add_input_value(
        "y2_shape_before_transpose", Tensor(dtype=Int(), shape=(4,), device=devices[0])
    )
    y2_shape_after_transpose = function.add_input_value(
        "y2_shape_after_transpose", Tensor(dtype=Int(), shape=(3,), device=devices[0])
    )

    x1 = function.add_op("MatMul", inputs=[x, w_1], output_names=["x1"])

    x2 = function.add_op("Reshape", inputs=[x1, x2_shape], output_names=["x2"])
    qkv = function.add_op(
        "Split",
        inputs=[x2],
        attributes={"num_splits": 3, "dim": 3},
        output_names=["qkv"],
    )
    q = function.add_op(
        "Select", inputs=[qkv], attributes={"dim": 0}, output_names=["q"]
    )
    k = function.add_op(
        "Select", inputs=[qkv], attributes={"dim": 1}, output_names=["k"]
    )
    v = function.add_op(
        "Select", inputs=[qkv], attributes={"dim": 2}, output_names=["v"]
    )

    q = function.add_op("Reshape", inputs=[q, qkv_shape], output_names=["q"])
    q = function.add_op(
        "Transpose", inputs=[q], attributes={"perm": (1, 0, 2)}, output_names=["q"]
    )
    k = function.add_op("Reshape", inputs=[k, qkv_shape], output_names=["k"])
    k = function.add_op(
        "Transpose", inputs=[k], attributes={"perm": (1, 2, 0)}, output_names=["k"]
    )
    v = function.add_op("Reshape", inputs=[v, qkv_shape], output_names=["v"])
    v = function.add_op(
        "Transpose", inputs=[v], attributes={"perm": (1, 0, 2)}, output_names=["v"]
    )

    y1 = function.add_op("MatMul", inputs=[q, k], output_names=["y1"])
    y2 = function.add_op("MatMul", inputs=[y1, v], output_names=["y2"])

    y2 = function.add_op(
        "Reshape", inputs=[y2, y2_shape_before_transpose], output_names=["y2"]
    )
    y2 = function.add_op(
        "Transpose", inputs=[y2], attributes={"perm": (2, 0, 1, 3)}, output_names=["y2"]
    )
    y2 = function.add_op(
        "Reshape", inputs=[y2, y2_shape_after_transpose], output_names=["y2"]
    )

    z = function.add_op("MatMul", inputs=[y2, w_2], output_names=["z"])

    function = function.finalize()
    ex = SequentialExecutor("numpy")
    _x = np.ones((s, b, h))
    _w_1 = np.ones((h, 3 * h))
    _w_2 = np.ones((h, h))
    _x2_shape = [s, b, n, 3 * h // n]
    _qkv_shape = [s, b * n, h // n]
    _y2_shape_before_transpose = [b, n, s, h // n]
    _y2_shape_after_transpose = [s, b, h]

    input_data = {
        function.inputs[0]: _x,
        function.inputs[1]: _w_1,
        function.inputs[2]: _w_2,
        function.inputs[3]: _x2_shape,
        function.inputs[4]: _qkv_shape,
        function.inputs[5]: _y2_shape_before_transpose,
        function.inputs[6]: _y2_shape_after_transpose,
    }
    res_before = ex.compute(function, input_data)
    np.testing.assert_array_almost_equal(
        res_before[function.outputs[0]],
        self_attention_before(n, _w_1, _w_2, _x),
    )

    # Transformed function
    transformed_function = shard_transform(
        function=function,
        ops=function.ops,
        input_dims={function.inputs[1]: 1, function.inputs[2]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "Allreduce"},
        },
        devices=devices[1:],
    )

    _x2_shape = [s, b, n // p, 3 * h // n]
    _qkv_shape = [s, b * n // p, h // n]
    _y2_shape_before_transpose = [b, n // p, s, h // n]
    _y2_shape_after_transpose = [s, b, h // p]
    input_data = {
        transformed_function.inputs[0]: _x,
        transformed_function.inputs[1]: _w_1,
        transformed_function.inputs[2]: _w_2,
        transformed_function.inputs[3]: _x2_shape,
        transformed_function.inputs[4]: _qkv_shape,
        transformed_function.inputs[5]: _y2_shape_before_transpose,
        transformed_function.inputs[6]: _y2_shape_after_transpose,
    }

    res_after = ex.compute(transformed_function, input_data)
    np.testing.assert_array_almost_equal(
        res_before[function.outputs[0]], res_after[transformed_function.outputs[0]][0]
    )
