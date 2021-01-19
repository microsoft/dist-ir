import numpy as np

from dist_ir.ir import Device, FunctionMaker
from dist_ir.ir.type import Float, Tensor
from dist_ir.transforms import apply_shard_transform
from dist_ir.executor import SequentialExecutor


def test_single_variable_data_parallel():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    a = function.add_input_value("a", Tensor(Float(), (4, 4)))
    b = function.add_input_value("b", Tensor(Float(), (4, 4)))
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    function = function.finalize()
    transformed_function = apply_shard_transform(
        function=function,
        ops=function.ops,
        input_dims={function.inputs[0]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "Gather", "dim": 0, "device": d0}
        },
        devices=[d0, d1],
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
    _a = np.ones((4, 4))
    _b = np.ones((4, 4))
    orig_res = ex.compute(function, {function.inputs[0]: _a, function.inputs[1]: _b})

    transformed_res = ex.compute(
        transformed_function,
        {transformed_function.inputs[0]: _a, transformed_function.inputs[1]: _b},
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


def test_double_variable_data_parallel():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    a = function.add_input_value("a", Tensor(Float(), (4, 4)))
    b = function.add_input_value("b", Tensor(Float(), (4, 4)))
    c = function.add_input_value("c", Tensor(Float(), (4, 4)))
    x = function.add_op("MatMul", "MatMul", inputs=[a, b], output_names=["x"])
    y = function.add_op("Add", "Add", inputs=[x, c], output_names=["y"])
    function = function.finalize()
    transformed_function = apply_shard_transform(
        function=function,
        ops=function.ops,
        input_dims={function.inputs[0]: 0, function.inputs[2]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "Gather", "dim": 0, "device": d0}
        },
        devices=[d0, d1],
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
    _a = np.ones((4, 4))
    _b = np.ones((4, 4))
    _c = np.ones((4, 4))
    orig_res = ex.compute(
        function,
        {function.inputs[0]: _a, function.inputs[1]: _b, function.inputs[2]: _c},
    )

    transformed_res = ex.compute(
        transformed_function,
        {
            transformed_function.inputs[0]: _a,
            transformed_function.inputs[1]: _b,
            transformed_function.inputs[2]: _c,
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


def test_single_variable_horizontal_parallel():
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
    transformed_function = apply_shard_transform(
        function=function,
        ops=[function.ops[0]],
        input_dims={function.inputs[1]: 1},
        reduction_params={
            function.ops[0].outputs[0]: {"op_type": "Gather", "dim": 1, "device": d0}
        },
        devices=[d0, d1],
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


def test_double_variable_horizontal_parallel():
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

    transformed_function = apply_shard_transform(
        function=function,
        ops=function.ops,
        input_dims={function.inputs[1]: 1, function.inputs[2]: 0},
        reduction_params={function.outputs[0]: {"op_type": "Allreduce", "device": d0}},
        devices=[d0, d1],
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


def test_mnist_data_parallel():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    batch_size = 16
    x = function.add_input_value("x", Tensor(Float(), (batch_size, 4)))
    z = function.add_input_value("z", Tensor(Float(), (batch_size, 1)))
    wA = function.add_input_value("wA", Tensor(Float(), (4, 2)))
    wB = function.add_input_value("wB", Tensor(Float(), (2, 1)))
    a = function.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = function.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    l = function.add_op(
        "Loss", "Loss", inputs=[y, z], attributes={"N": batch_size}, output_names=["l"]
    )
    dl = function.add_op(
        "LossGrad",
        "LossGrad",
        inputs=[y, z],
        attributes={"N": batch_size},
        output_names=["dl"],
    )
    da, dwB = function.add_op(
        "MatMulGrad", "MatMul1Grad", inputs=[a, wB, dl], output_names=["da", "dwB"]
    )
    dx, dwA = function.add_op(
        "MatMulGrad", "MatMul0Grad", inputs=[x, wA, da], output_names=["dx", "dwA"]
    )
    function = function.finalize()
    transformed_function = apply_shard_transform(
        function=function,
        ops=function.ops,
        input_dims={function.inputs[0]: 0, function.inputs[1]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "Gather", "dim": 0, "device": d0},
            function.outputs[1]: {"op_type": "Allreduce"},
            function.outputs[2]: {"op_type": "Gather", "dim": 0, "device": d0},
            function.outputs[3]: {"op_type": "Allreduce"},
        },
        devices=[d0, d1],
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
    _x = np.arange(batch_size * 4).reshape((batch_size, 4))
    _z = np.ones((batch_size, 1))
    _wA = np.ones((4, 2))
    _wB = np.ones((2, 1))
    orig_res = ex.compute(
        function,
        {
            function.inputs[0]: _x,
            function.inputs[1]: _z,
            function.inputs[2]: _wA,
            function.inputs[3]: _wB,
        },
    )

    transformed_res = ex.compute(
        transformed_function,
        {
            transformed_function.inputs[0]: _x,
            transformed_function.inputs[1]: _z,
            transformed_function.inputs[2]: _wA,
            transformed_function.inputs[3]: _wB,
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
    np.testing.assert_array_almost_equal(
        orig_res[function.outputs[1]],
        transformed_res[transformed_function.outputs[1]][0],
    )
    np.testing.assert_array_almost_equal(
        orig_res[function.outputs[2]],
        transformed_res[transformed_function.outputs[2]],
    )
    np.testing.assert_array_almost_equal(
        orig_res[function.outputs[3]],
        transformed_res[transformed_function.outputs[3]][0],
    )
