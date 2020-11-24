from dist_ir.ir import Module
from dist_ir.ir import Topology
from dist_ir.ir.type import Float
from dist_ir.ir.type import Tensor
from dist_ir.executor import DistributedSimulator
from dist_ir.executor.shape_inference import infer_shapes
from dist_ir.transforms import DataParallelTransform


def test_single_device():
    module = Module()
    topology = Topology()

    d = topology.add_device("gpu")

    a = module.add_input_value("a", Tensor(Float(), (4, 4)), device=d)
    b = module.add_input_value("b", Tensor(Float(), (4, 4)), device=d)
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], device=d)
    infer_shapes(module)

    per_op_costs = {"MatMul": {"gpu": 10}}

    simulator = DistributedSimulator(topology, per_op_costs)
    timestamps, peak_memory = simulator.simulate(module)
    assert timestamps[d] == 10.0
    assert peak_memory[d] == 192.0


def test_double_device():
    module = Module()
    topology = Topology()

    d0 = topology.add_device("gpu")
    d1 = topology.add_device("gpu")

    topology.set_bandwidth(d0, d1, 100)

    a = module.add_input_value("a", Tensor(Float(), (4, 4)), device=d0)
    b = module.add_input_value("b", Tensor(Float(), (4, 4)), device=d0)
    c = module.add_input_value("c", Tensor(Float(), (4, 4)), device=d1)
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], device=d0)
    y = module.add_op("MatMul", "MatMul1", inputs=[x, c], device=d1)
    infer_shapes(module)

    per_op_costs = {"MatMul": {"gpu": 10}}

    simulator = DistributedSimulator(topology, per_op_costs)
    timestamps, peak_memory = simulator.simulate(module)
    assert timestamps[d0] == 10.64
    assert timestamps[d1] == 20.64
    assert peak_memory[d0] == 192.0
    assert peak_memory[d1] == 192.0


def test_sync():
    module = Module()
    topology = Topology()

    d0 = topology.add_device("gpu")
    d1 = topology.add_device("gpu")
    d2 = topology.add_device("gpu")

    topology.set_bandwidth(d0, d1, 100)
    topology.set_bandwidth(d0, d2, 50)
    topology.set_bandwidth(d1, d2, 50)

    a = module.add_input_value("a", Tensor(Float(), (4, 4)), device=d0)
    b = module.add_input_value("b", Tensor(Float(), (4, 4)), device=d0)
    c = module.add_input_value("c", Tensor(Float(), (4, 4)), device=d1)
    d = module.add_input_value("d", Tensor(Float(), (4, 4)), device=d1)
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], device=d0)
    y = module.add_op("MatMul", "MatMul1", inputs=[c, d], device=d1)
    z = module.add_op("Allreduce", "Allreduce0", inputs=[x, y], device=d2)
    infer_shapes(module)

    per_op_costs = {"Allreduce": {"gpu": 0}, "MatMul": {"gpu": 10}}

    simulator = DistributedSimulator(topology, per_op_costs)
    timestamps, peak_memory = simulator.simulate(module)
    assert timestamps[d0] == 11.28
    assert timestamps[d1] == 11.28
    assert timestamps[d2] == 11.28
    assert peak_memory[d0] == 192.0
    assert peak_memory[d1] == 192.0
    assert peak_memory[d2] == 192.0


def test_data_parallel():
    module = Module()
    topology = Topology()

    d0 = topology.add_device("gpu")
    d1 = topology.add_device("gpu")
    topology.set_bandwidth(d0, d1, 100)

    a = module.add_input_value("a", Tensor(Float(), (4, 4)), device=d0)
    b = module.add_input_value("b", Tensor(Float(), (4, 4)), device=d0)
    c = module.add_input_value("c", Tensor(Float(), (4, 4)), device=d0)
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], output_names=["x"], device=d0)
    y = module.add_op("MatMul", "MatMul1", inputs=[x, c], output_names=["y"], device=d0)
    transform = DataParallelTransform(partition_map={"a": 0}, devices=[d0, d1])
    transformed_module = transform.apply(module)

    infer_shapes(transformed_module)

    print(transformed_module)

    per_op_costs = {
        "Allreduce": {"gpu": 0},
        "Broadcast": {"gpu": 0},
        "MatMul": {"gpu": 10},
        "Scatter": {"gpu": 0},
    }
    simulator = DistributedSimulator(topology, per_op_costs)
    timestamps, peak_memory = simulator.simulate(transformed_module)
    assert timestamps[d0] == 30.64
    assert timestamps[d1] == 30.64
    assert peak_memory[d0] == 352.0
    assert peak_memory[d1] == 160


if __name__ == "__main__":
    test_data_parallel()
