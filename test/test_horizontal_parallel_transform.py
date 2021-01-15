import numpy as np

from dist_ir.ir import Device, FunctionMaker
from dist_ir.ir.type import Float, Tensor
from dist_ir.transforms import ParallelMapTransform
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
    transform = ParallelMapTransform(
        ops=[function.ops[0]],
        input_dims={function.inputs[1]: 1},
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

    transform = ParallelMapTransform(
        ops=function.ops,
        input_dims={function.inputs[1]: 1, function.inputs[2]: 0},
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
    transform = ParallelMapTransform(
        ops=function.ops,
        input_dims={function.inputs[0]: 0},
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
