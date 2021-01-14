import numpy as np

from dist_ir.ir import Device, FunctionMaker
from dist_ir.ir.type import Float, Tensor
from dist_ir.transforms import HorizontalParallelTransform
from dist_ir.executor import SequentialExecutor
from dist_ir.executor.type_inference import infer_types


def test_single_variable_partition():
    batch_size = 16
    input_dim = 32
    hidden_dim = 16
    output_dim = 8

    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = function.add_input_value("x", Tensor(Float(), (batch_size, input_dim)))
    wA = function.add_input_value("wA", Tensor(Float(), (input_dim, hidden_dim)))
    wB = function.add_input_value("wB", Tensor(Float(), (hidden_dim, output_dim)))
    a = function.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = function.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    function = function.finalize()
    transform = HorizontalParallelTransform(
        ops=[function.ops[0]],
        param_dims={function.inputs[1]: 1},
        reduction_params={
            function.ops[0].outputs[0]: {"op_type": "Gather", "dim": 1, "device": d0}
        },
        devices=[d0, d1],
    )
    transformed_function = transform.apply(function)

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    print(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    print(transformed_function)

    ex = SequentialExecutor("numpy")
    _x = np.random.normal(size=(batch_size, input_dim))
    _wA = np.random.normal(size=(input_dim, hidden_dim))
    _wB = np.random.normal(size=(hidden_dim, output_dim))
    orig_res = ex.compute(
        function,
        {function.inputs[0]: _x, function.inputs[1]: _wA, function.inputs[2]: _wB},
    )

    transformed_res = ex.compute(
        transformed_function,
        {
            transformed_function.inputs[0]: _x,
            transformed_function.inputs[1]: _wA,
            transformed_function.inputs[2]: _wB,
        },
    )

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    for k, v in orig_res.items():
        print(k)
        print(v)
        print()
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    for k, v in transformed_res.items():
        print(k)
        print(v)
        print()

    np.testing.assert_array_almost_equal(
        orig_res[function.outputs[0]], transformed_res[transformed_function.outputs[0]]
    )


def test_double_variable_partition():
    batch_size = 16
    input_dim = 32
    hidden_dim = 16
    output_dim = 8

    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = function.add_input_value("x", Tensor(Float(), (batch_size, input_dim)))
    wA = function.add_input_value("wA", Tensor(Float(), (input_dim, hidden_dim)))
    wB = function.add_input_value("wB", Tensor(Float(), (hidden_dim, output_dim)))
    a = function.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = function.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    function = function.finalize()

    transform = HorizontalParallelTransform(
        ops=function.ops,
        param_dims={function.inputs[1]: 1, function.inputs[2]: 0},
        reduction_params={function.outputs[0]: {"op_type": "Allreduce", "device": d0}},
        devices=[d0, d1],
    )
    transformed_function = transform.apply(function)

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    print(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    print(transformed_function)

    ex = SequentialExecutor("numpy")
    _x = np.random.normal(size=(batch_size, input_dim))
    _wA = np.random.normal(size=(input_dim, hidden_dim))
    _wB = np.random.normal(size=(hidden_dim, output_dim))
    orig_res = ex.compute(
        function,
        {function.inputs[0]: _x, function.inputs[1]: _wA, function.inputs[2]: _wB},
    )
    transformed_res = ex.compute(
        transformed_function,
        {
            transformed_function.inputs[0]: _x,
            transformed_function.inputs[1]: _wA,
            transformed_function.inputs[2]: _wB,
        },
    )

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    for k, v in orig_res.items():
        print(k)
        print(v)
        print()
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    for k, v in transformed_res.items():
        print(k)
        print(v)
        print()

    np.testing.assert_array_almost_equal(
        orig_res[function.outputs[0]],
        transformed_res[transformed_function.outputs[0]][0],
    )


def test_data_parallel():
    batch_size = 16
    input_dim = 32
    hidden_dim = 16
    output_dim = 8

    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = function.add_input_value("x", Tensor(Float(), (batch_size, input_dim)))
    wA = function.add_input_value("wA", Tensor(Float(), (input_dim, hidden_dim)))
    wB = function.add_input_value("wB", Tensor(Float(), (hidden_dim, output_dim)))
    a = function.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = function.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    function = function.finalize()
    transform = HorizontalParallelTransform(
        ops=function.ops,
        param_dims={function.inputs[0]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "Gather", "dim": 0, "device": d0}
        },
        devices=[d0, d1],
    )
    transformed_function = transform.apply(function)

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    print(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    print(transformed_function)

    ex = SequentialExecutor("numpy")
    _x = np.random.normal(size=(batch_size, input_dim))
    _wA = np.random.normal(size=(input_dim, hidden_dim))
    _wB = np.random.normal(size=(hidden_dim, output_dim))
    orig_res = ex.compute(
        function,
        {function.inputs[0]: _x, function.inputs[1]: _wA, function.inputs[2]: _wB},
    )
    transformed_res = ex.compute(
        transformed_function,
        {
            transformed_function.inputs[0]: _x,
            transformed_function.inputs[1]: _wA,
            transformed_function.inputs[2]: _wB,
        },
    )

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    for k, v in orig_res.items():
        print(k)
        print(v)
        print()
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    for k, v in transformed_res.items():
        print(k)
        print(v)
        print()

    np.testing.assert_array_almost_equal(
        orig_res[function.outputs[0]], transformed_res[transformed_function.outputs[0]]
    )


def test_attention_debug():
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

    q = function.add_op("MatMul", inputs=[Q, W_Q], output_names=["q"])
    k = function.add_op("MatMul", inputs=[K, W_K], output_names=["k"])
    v = function.add_op("MatMul", inputs=[V, W_V], output_names=["v"])

    k_t = function.add_op("Transpose", inputs=[k], output_names=["k_t"])
    qk_t = function.add_op("MatMul", inputs=[q, k_t], output_names=["qk_t"])
    qk_tv = function.add_op("MatMul", inputs=[qk_t, v], output_names=["qk_tv"])
    function = function.finalize()
    function = infer_types(function, function.inputs)

    transform = HorizontalParallelTransform(
        ops=function.ops[3:],
        param_dims={
            function.ops[0].outputs[0]: 1,
            function.ops[1].outputs[0]: 1,
            function.ops[2].outputs[0]: 1,
        },
        reduction_params={
            function.outputs[0]: {
                "op_type": "Gather",
                "dim": 1,
                "device": devices[0],
            }
        },
        devices=devices[1:],
    )

    transformed_function = transform.apply(function)
    transformed_function = infer_types(
        transformed_function, transformed_function.inputs
    )

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    print(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    print(transformed_function)

    ex = SequentialExecutor("numpy")
    _Q = np.random.normal(size=(n, d_model))
    _K = np.random.normal(size=(n, d_model))
    _V = np.random.normal(size=(n, d_model))
    _W_Q = np.random.normal(size=(d_model, d_model))
    _W_K = np.random.normal(size=(d_model, d_model))
    _W_V = np.random.normal(size=(d_model, d_model))

    orig_res = ex.compute(
        function,
        {
            function.inputs[0]: _Q,
            function.inputs[1]: _K,
            function.inputs[2]: _V,
            function.inputs[3]: _W_Q,
            function.inputs[4]: _W_K,
            function.inputs[5]: _W_V,
        },
    )
    transformed_res = ex.compute(
        transformed_function,
        {
            transformed_function.inputs[0]: _Q,
            transformed_function.inputs[1]: _K,
            transformed_function.inputs[2]: _V,
            transformed_function.inputs[3]: _W_Q,
            transformed_function.inputs[4]: _W_K,
            transformed_function.inputs[5]: _W_V,
        },
    )

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    for k, v in orig_res.items():
        print(k)
        print(v)
        print()
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    for k, v in transformed_res.items():
        print(k)
        print(v)
        print()

    np.testing.assert_array_almost_equal(
        orig_res[function.outputs[0]], transformed_res[transformed_function.outputs[0]]
    )


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

    q = function.add_op("MatMul", inputs=[Q, W_Q], output_names=["q"])
    k = function.add_op("MatMul", inputs=[K, W_K], output_names=["k"])
    v = function.add_op("MatMul", inputs=[V, W_V], output_names=["v"])

    k_t = function.add_op("Transpose", inputs=[k], output_names=["k_t"])
    qk_t = function.add_op("MatMul", inputs=[q, k_t], output_names=["qk_t"])
    qk_tv = function.add_op("MatMul", inputs=[qk_t, v], output_names=["qk_tv"])
    a = function.add_op("MatMul", inputs=[qk_tv, W_O], output_names=["a"])
    function = function.finalize()
    function = infer_types(function, function.inputs)

    transform = HorizontalParallelTransform(
        ops=function.ops[:-1],
        param_dims={
            function.inputs[3]: 1,
            function.inputs[4]: 1,
            function.inputs[5]: 1,
        },
        reduction_params={
            function.ops[-2].outputs[0]: {
                "op_type": "Gather",
                "dim": 1,
                "device": devices[0],
            }
        },
        devices=devices[1:],
    )

    transformed_function = transform.apply(function)
    transformed_function = infer_types(
        transformed_function, transformed_function.inputs
    )

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    print(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    print(transformed_function)

    ex = SequentialExecutor("numpy")
    _Q = np.random.normal(size=(n, d_model))
    _K = np.random.normal(size=(n, d_model))
    _V = np.random.normal(size=(n, d_model))
    _W_Q = np.random.normal(size=(d_model, d_model))
    _W_K = np.random.normal(size=(d_model, d_model))
    _W_V = np.random.normal(size=(d_model, d_model))
    _W_O = np.random.normal(size=(d_model, d_model))

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
        },
    )
    transformed_res = ex.compute(
        transformed_function,
        {
            transformed_function.inputs[0]: _Q,
            transformed_function.inputs[1]: _K,
            transformed_function.inputs[2]: _V,
            transformed_function.inputs[3]: _W_Q,
            transformed_function.inputs[4]: _W_K,
            transformed_function.inputs[5]: _W_V,
            transformed_function.inputs[6]: _W_O,
        },
    )

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    for k, v in orig_res.items():
        print(k)
        print(v)
        print()
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    for k, v in transformed_res.items():
        print(k)
        print(v)
        print()

    np.testing.assert_array_almost_equal(
        orig_res[function.outputs[0]], transformed_res[transformed_function.outputs[0]]
    )


if __name__ == "__main__":
    # test_double_variable_partition()
    # test_data_parallel()
    test_attention_debug()
