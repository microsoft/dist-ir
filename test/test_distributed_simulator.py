from dist_ir.ir import Module
from dist_ir.ir import Topology
from dist_ir.ir.type import Float
from dist_ir.ir.type import Tensor
from dist_ir.executor import DistributedSimulator
from dist_ir.executor.shape_inference import infer_shapes


def test_single_matmul():
    module = Module()
    topology = Topology()

    d = topology.add_device("gpu")

    a = module.add_input_value("a", Tensor(Float(), (4, 4)), device=d)
    b = module.add_input_value("b", Tensor(Float(), (4, 4)), device=d)
    x = module.add_op("MatMul", "MatMul0", inputs=[a, b], device=d)
    infer_shapes(module)

    per_op_costs = {"MatMul": {"gpu": 10}}

    simulator = DistributedSimulator(topology, per_op_costs)
    simulator.simulate(module)


if __name__ == "__main__":
    test_single_matmul()
