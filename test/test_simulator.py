from dist_ir.ir import cpprint, FunctionMaker, Topology
from dist_ir.ir.type import Float32
from dist_ir.ir.type import Tensor
from dist_ir.executor.cost_model import CostModel
from dist_ir.executor.type_inference import infer_types
from dist_ir.executor import Simulator
from dist_ir.transforms import shard_transform


def test_single_device():
    function = FunctionMaker()
    topology = Topology()
    d = topology.add_device("gpu")
    simulator = Simulator(CostModel(topology))

    a = function.add_input_value("a", None)
    b = function.add_input_value("b", None)
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b])
    function = function.finalize()

    inputs = (Tensor(dtype=Float32(), shape=(400, 400), device=d),) * 2
    state1 = simulator.simulate(function, inputs)
    assert d in state1.timestamps
    assert d in state1.peak_memory

    inputs = (Tensor(dtype=Float32(), shape=(800, 800), device=d),) * 2
    state2 = simulator.simulate(function, inputs)
    assert d in state2.timestamps
    assert d in state2.peak_memory
    assert state1.timestamps[d] < state2.timestamps[d]
    # TODO: Check specific values


# Disable test until we fix Pmap device assignment for simulation
# TODO remove pmap, or add simulator support for pmap
def _test_data_parallel():
    function = FunctionMaker()
    topology = Topology()

    d0 = topology.add_device("gpu")
    d1 = topology.add_device("gpu")
    topology.set_bandwidth(d0, d1, 2)

    a = function.add_input_value("a", Tensor(Float32(), (4, 4), device=d0))
    b = function.add_input_value("b", Tensor(Float32(), (4, 4), device=d0))
    c = function.add_input_value("c", Tensor(Float32(), (4, 4), device=d0))
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    y = function.add_op("MatMul", "MatMul1", inputs=[x, c], output_names=["y"])
    function = function.finalize()
    function = infer_types(function, [a, b, c])
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

    cpprint(transformed_function)
    simulator = Simulator(CostModel(topology))
    simulator_state = simulator.simulate(
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
    d = topology.add_device("gpu")
    simulator = Simulator(CostModel(topology))

    a = function.add_input_value("a", None)
    b = function.add_input_value("b", None)
    x = function.add_op("MatMul", "MatMul0", inputs=[a, b])
    function = function.finalize()

    inputs = (Tensor(dtype=Float32(), shape=(400, 400), device=d),) * 2
    state = simulator.simulate(function, inputs)
    state.dump_chrome_trace("test/trace.json")


if __name__ == "__main__":
    test_single_device()
