from dist_ir.ir import FunctionMaker
from dist_ir.ir import Topology
from dist_ir.ir.type import Float
from dist_ir.ir.type import Tensor
from dist_ir.executor.cost_inference import CostModel
from dist_ir.executor import DistributedSimulator
from dist_ir.transforms import DataParallelTransform


def test_single_device():
    function = FunctionMaker()
    topology = Topology()

    d = topology.add_device("gpu")

    a = function.add_input_value("a", Tensor(dtype=Float(), shape=(4, 4), device=d))
    b = function.add_input_value("b", Tensor(dtype=Float(), shape=(4, 4), device=d))
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b])
    function = function.finalize()
    device_speeds = {"gpu": 1.0e13}
    cost_model = CostModel(topology, device_speeds)
    simulator = DistributedSimulator(cost_model)
    simulator_state = simulator.simulate(function)
    assert d in simulator_state.timestamps
    assert d in simulator_state.peak_memory
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
    transform = DataParallelTransform(
        batch_dims={"a": 0},
        reduction_params={"y": {"op_type": "Gather", "dim": 0, "device": d0}},
        devices=[d0, d1],
    )
    transformed_function = transform.apply(function)

    print(transformed_function)
    device_speeds = {"gpu": 1.0e13}
    cost_model = CostModel(topology, device_speeds)
    simulator = DistributedSimulator(cost_model)
    simulator_state = simulator.simulate(transformed_function)
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

    device_speeds = {"gpu": 1.0e13}
    cost_model = CostModel(topology, device_speeds)
    simulator = DistributedSimulator(cost_model)

    transform = DataParallelTransform(
        batch_dims={"a": 0},
        reduction_params={"y": {"op_type": "Gather", "dim": 0, "device": d0}},
        devices=[d0, d1],
    )
    transformed_function = transform.apply(function)

    simulation = simulator.simulate(transformed_function)
    simulation.dump_chrome_trace("test/trace.json")
