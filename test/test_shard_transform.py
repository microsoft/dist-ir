import numpy as np

from dist_ir.ir import cpprint, Device, FunctionMaker
from dist_ir.ir.type import Float32, Tensor
from dist_ir.transforms import shard_transform
from dist_ir.executor import SequentialExecutor, infer_types


def test_single_variable_data_parallel():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    a = function.add_input_value("a", Tensor(Float32(), (4, 4)))
    b = function.add_input_value("b", Tensor(Float32(), (4, 4)))
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    function = function.finalize()
    function = infer_types(function, function.inputs)
    transformed_function = shard_transform(
        function=function,
        ops=function.ops,
        input_dims={function.inputs[0]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "MPIGather", "axis": 0, "device": d0}
        },
        devices=[d0, d1],
    )
    transformed_function = infer_types(
        transformed_function, transformed_function.inputs
    )

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    cpprint(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    cpprint(transformed_function)

    ex = SequentialExecutor("numpy")
    _a = np.ones((4, 4))
    _b = np.ones((4, 4))
    orig_res = ex.compute(function, [_a, _b])

    transformed_res = ex.compute(transformed_function, [_a, _b])

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    print(orig_res)
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    print(transformed_res)

    np.testing.assert_array_almost_equal(orig_res[0], transformed_res[0])


def test_double_variable_data_parallel():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    a = function.add_input_value("a", Tensor(Float32(), (4, 4)))
    b = function.add_input_value("b", Tensor(Float32(), (4, 4)))
    c = function.add_input_value("c", Tensor(Float32(), (4, 4)))
    x = function.add_op("MatMul", "MatMul", inputs=[a, b], output_names=["x"])
    y = function.add_op("Add", "Add", inputs=[x, c], output_names=["y"])
    function = function.finalize()
    transformed_function = shard_transform(
        function=function,
        ops=function.ops,
        input_dims={function.inputs[0]: 0, function.inputs[2]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "MPIGather", "axis": 0, "device": d0}
        },
        devices=[d0, d1],
    )

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    cpprint(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    cpprint(transformed_function)

    ex = SequentialExecutor("numpy")
    _a = np.ones((4, 4))
    _b = np.ones((4, 4))
    _c = np.ones((4, 4))
    orig_res = ex.compute(function, [_a, _b, _c])

    transformed_res = ex.compute(transformed_function, [_a, _b, _c])

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    print(orig_res)
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    print(transformed_res)
    print()

    np.testing.assert_array_almost_equal(orig_res[0], transformed_res[0])


def test_single_variable_horizontal_parallel():
    batch_size = 16
    input_dim = 32
    hidden_dim = 16
    output_dim = 8

    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = function.add_input_value("x", Tensor(Float32(), (batch_size, input_dim)))
    wA = function.add_input_value("wA", Tensor(Float32(), (input_dim, hidden_dim)))
    wB = function.add_input_value("wB", Tensor(Float32(), (hidden_dim, output_dim)))
    a = function.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = function.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    function = function.finalize()
    transformed_function = shard_transform(
        function=function,
        ops=[function.ops[0]],
        input_dims={function.inputs[1]: 1},
        reduction_params={
            function.ops[0].outputs[0]: {"op_type": "MPIGather", "axis": 1, "device": d0}
        },
        devices=[d0, d1],
    )

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    cpprint(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    cpprint(transformed_function)

    ex = SequentialExecutor("numpy")
    _x = np.random.normal(size=(batch_size, input_dim))
    _wA = np.random.normal(size=(input_dim, hidden_dim))
    _wB = np.random.normal(size=(hidden_dim, output_dim))
    orig_res = ex.compute(function, [_x, _wA, _wB])

    transformed_res = ex.compute(transformed_function, [_x, _wA, _wB])

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    print(orig_res)
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    print(transformed_res)
    print()

    np.testing.assert_array_almost_equal(orig_res[0], transformed_res[0])


def test_double_variable_horizontal_parallel():
    batch_size = 16
    input_dim = 32
    hidden_dim = 16
    output_dim = 8

    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    x = function.add_input_value("x", Tensor(Float32(), (batch_size, input_dim)))
    wA = function.add_input_value("wA", Tensor(Float32(), (input_dim, hidden_dim)))
    wB = function.add_input_value("wB", Tensor(Float32(), (hidden_dim, output_dim)))
    a = function.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = function.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    function = function.finalize()

    transformed_function = shard_transform(
        function=function,
        ops=function.ops,
        input_dims={function.inputs[1]: 1, function.inputs[2]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "MPIAllreduce", "device": d0}
        },
        devices=[d0, d1],
    )

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    cpprint(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    cpprint(transformed_function)

    ex = SequentialExecutor("numpy")
    _x = np.random.normal(size=(batch_size, input_dim))
    _wA = np.random.normal(size=(input_dim, hidden_dim))
    _wB = np.random.normal(size=(hidden_dim, output_dim))
    orig_res = ex.compute(function, [_x, _wA, _wB])
    transformed_res = ex.compute(transformed_function, [_x, _wA, _wB])

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    print(orig_res)
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    print(transformed_res)
    print()

    np.testing.assert_array_almost_equal(orig_res[0], transformed_res[0][0])


def test_mnist_data_parallel():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")

    batch_size = 16
    x = function.add_input_value("x", Tensor(Float32(), (batch_size, 4)))
    z = function.add_input_value("z", Tensor(Float32(), (batch_size, 1)))
    wA = function.add_input_value("wA", Tensor(Float32(), (4, 2)))
    wB = function.add_input_value("wB", Tensor(Float32(), (2, 1)))
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
    transformed_function = shard_transform(
        function=function,
        ops=function.ops,
        input_dims={function.inputs[0]: 0, function.inputs[1]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "MPIGather", "axis": 0, "device": d0},
            function.outputs[1]: {"op_type": "MPIAllreduce"},
            function.outputs[2]: {"op_type": "MPIGather", "axis": 0, "device": d0},
            function.outputs[3]: {"op_type": "MPIAllreduce"},
        },
        devices=[d0, d1],
    )

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    cpprint(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    cpprint(transformed_function)

    ex = SequentialExecutor("numpy")
    _x = np.arange(batch_size * 4).reshape((batch_size, 4))
    _z = np.ones((batch_size, 1))
    _wA = np.ones((4, 2))
    _wB = np.ones((2, 1))
    orig_res = ex.compute(function, [_x, _z, _wA, _wB])

    transformed_res = ex.compute(transformed_function, [_x, _z, _wA, _wB])

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    print(orig_res)
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    print(transformed_res)
    print()

    np.testing.assert_array_almost_equal(orig_res[0], transformed_res[0])
    np.testing.assert_array_almost_equal(orig_res[1], transformed_res[1][0])
    np.testing.assert_array_almost_equal(orig_res[2], transformed_res[2])
    np.testing.assert_array_almost_equal(orig_res[3], transformed_res[3][0])
