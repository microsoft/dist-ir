from dist_ir.ir import cpprint, FunctionMaker, Topology
from dist_ir.ir.type import Float
from dist_ir.ir.type import Tensor
from dist_ir.executor.cost_model import CostModel
from dist_ir.executor.type_inference import infer_types
from dist_ir.executor import Simulator
from dist_ir.transforms import shard_transform


def test_single_device():
    function = FunctionMaker()
    topology = Topology()

    d = topology.add_device("gpu")

    a = function.add_input_value("a", Tensor(dtype=Float(), shape=(4, 4), device=d))
    b = function.add_input_value("b", Tensor(dtype=Float(), shape=(4, 4), device=d))
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b])
    function = function.finalize()
    function = infer_types(function, [a, b])
    device_speeds = {"gpu": 1.0e13}
    # TODO shouldn't device_speeds be set in the topology?
    simulator = Simulator(CostModel(topology, device_speeds))
    state = simulator.interpret(function, (v.type for v in function.inputs))
    assert d in state.timestamps
    assert d in state.peak_memory
    # TODO: Check specific values


def test_data_parallel():
    function = FunctionMaker()
    topology = Topology()

    d0 = topology.add_device("gpu")
    d1 = topology.add_device("gpu")
    topology.set_bandwidth(d0, d1, 2)

    a = function.add_input_value("a", Tensor(Float(), (4, 4), device=d0))
    b = function.add_input_value("b", Tensor(Float(), (4, 4), device=d0))
    c = function.add_input_value("c", Tensor(Float(), (4, 4), device=d0))
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    y = function.add_op("MatMul", "MatMul1", inputs=[x, c], output_names=["y"])
    function = function.finalize()
    function = infer_types(function, [a, b, c])
    transformed_function = shard_transform(
        function=function,
        ops=function.ops,
        input_dims={function.inputs[0]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "Gather", "dim": 0, "device": d0}
        },
        devices=[d0, d1],
    )
    transformed_function = infer_types(
        transformed_function, transformed_function.inputs
    )

    cpprint(transformed_function)
    device_speeds = {"gpu": 1.0e13}
    simulator = Simulator(CostModel(topology, device_speeds))
    simulator_state = simulator.interpret(
        transformed_function, (v.type for v in transformed_function.inputs)
    )
    assert d0 in simulator_state.timestamps
    assert d1 in simulator_state.timestamps
    assert d0 in simulator_state.peak_memory
    assert d1 in simulator_state.peak_memory
    # TODO: Check specific values


def test_chrome_trace():
    function = FunctionMaker()

    topology = Topology()
    d0 = topology.add_device("gpu")
    d1 = topology.add_device("gpu")
    topology.set_bandwidth(d0, d1, 2)

    a = function.add_input_value("a", Tensor(Float(), (4, 4), device=d0))
    b = function.add_input_value("b", Tensor(Float(), (4, 4), device=d0))
    c = function.add_input_value("c", Tensor(Float(), (4, 4), device=d0))
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    y = function.add_op("MatMul", "MatMul1", inputs=[x, c], output_names=["y"])
    function = function.finalize()
    function = infer_types(function, [a, b, c])

    device_speeds = {"gpu": 1.0e13}
    simulator = Simulator(CostModel(topology, device_speeds))

    transformed_function = shard_transform(
        function=function,
        ops=function.ops,
        input_dims={function.inputs[0]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "Gather", "dim": 0, "device": d0}
        },
        devices=[d0, d1],
    )
    transformed_function = infer_types(
        transformed_function, transformed_function.inputs
    )

    simulation = simulator.interpret(
        transformed_function, (v.type for v in transformed_function.inputs)
    )
    simulation.dump_chrome_trace("test/trace.json")


def test_function_call():
    topology = Topology()
    d0 = topology.add_device("gpu")

    layer = FunctionMaker()
    x = layer.add_input_value("x", None)
    w = layer.add_input_value("w", None)
    _ = layer.add_op("MatMul", inputs=[x, w])
    layer = layer.finalize()
    fn = FunctionMaker()
    x = fn.add_input_value("x", Tensor(Float(), (4, 5), device=d0))
    w1 = fn.add_input_value("w1", Tensor(Float(), (5, 6), device=d0))
    w2 = fn.add_input_value("w2", Tensor(Float(), (6, 2), device=d0))
    a1 = fn.add_op("FnCall", inputs=[x, w1], subfunctions=[layer])
    _ = fn.add_op("FnCall", inputs=[a1, w2], subfunctions=[layer])
    fn = fn.finalize()
    fn = infer_types(fn, fn.inputs)

    device_speeds = {"gpu": 1.0e13}
    simulator = Simulator(CostModel(topology, device_speeds))
    simulation = simulator.interpret(fn, (v.type for v in fn.inputs))
    simulation.dump_chrome_trace("test/trace.json")
