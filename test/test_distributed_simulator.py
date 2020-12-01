from dist_ir.ir import Module
from dist_ir.ir import Topology
from dist_ir.ir.type import Float
from dist_ir.ir.type import Tensor
from dist_ir.executor.cost_inference import CostModel
from dist_ir.executor import DistributedSimulator
from dist_ir.transforms import DataParallelTransform


def test_single_device():
    module = Module()
    topology = Topology()

    d = topology.add_device("gpu")

    a = module.add_input_value("a", Tensor(dtype=Float(), shape=(4, 4), device=d))
    b = module.add_input_value("b", Tensor(dtype=Float(), shape=(4, 4), device=d))
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b])
    module.finalize()
    device_speeds = {"gpu": 1.0e13}
    cost_model = CostModel(topology, device_speeds)
    simulator = DistributedSimulator(cost_model)
    simulator_state = simulator.simulate(module)
    assert d in simulator_state.timestamps
    assert d in simulator_state.peak_memory
    # TODO: Check specific values


def test_data_parallel():
    module = Module()
    topology = Topology()

    d0 = topology.add_device("gpu")
    d1 = topology.add_device("gpu")
    topology.set_bandwidth(d0, d1, 2)

    a = module.add_input_value("a", Tensor(Float(), (4, 4), device=d0))
    b = module.add_input_value("b", Tensor(Float(), (4, 4), device=d0))
    c = module.add_input_value("c", Tensor(Float(), (4, 4), device=d0))
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"])
    y = module.add_op("MatMul", "MatMul1", inputs=[x, c], output_names=["y"])
    module.finalize()
    transform = DataParallelTransform(
        batch_dims={"a": 0},
        reduction_params={"y": {"op_type": "Gather"}},
        devices=[d0, d1],
    )
    transformed_module = transform.apply(module)

    transformed_module.finalize()
    print(transformed_module)
    device_speeds = {"gpu": 1.0e13}
    cost_model = CostModel(topology, device_speeds)
    simulator = DistributedSimulator(cost_model)
    simulator_state = simulator.simulate(transformed_module)
    assert d0 in simulator_state.timestamps
    assert d1 in simulator_state.timestamps
    assert d0 in simulator_state.peak_memory
    assert d1 in simulator_state.peak_memory
    # TODO: Check specific values


if __name__ == "__main__":
    test_data_parallel()
