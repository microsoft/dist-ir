import torch

from dist_ir.importer import import_from_onnx, parse_tensor_from_file
from dist_ir.ir import cpprint, Topology, Value
from dist_ir.ir.type import Float, Tensor
from dist_ir.executor.cost_model import CostModel
from dist_ir.executor import Simulator
from dist_ir.executor.type_inference import infer_types


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.matmul(x, x)
        return torch.relu(y)


def test_per_op_costs():
    module = MyModule()

    input_shape = (4, 4)
    onnx_file_path = "/tmp/mymodule.onnx"
    torch.onnx.export(module, (torch.randn(input_shape),), onnx_file_path, verbose=True)

    fn, _ = import_from_onnx(onnx_file_path)
    cpprint(fn)

    DEVICE_THROUGHPUT = 1.38e13  # FLOPS
    DRAM_BANDWIDTH = 7e11  # Bps
    topology = Topology()
    d0 = topology.add_device(
        "gpu", throughput=DEVICE_THROUGHPUT, dram_bandwidth=DRAM_BANDWIDTH
    )
    simulator = Simulator(CostModel(topology))

    # Type inference does shape propagation:
    input_values = [Value("x", Tensor(Float(), input_shape, d0))]
    fn = infer_types(fn, input_values)

    # Simulation results in a state which records the per-op costs:
    # simulation.per_op_costs: Op -> Device -> runtime
    simulation = simulator.interpret(fn, [v.type for v in input_values])
    print(simulation.per_op_costs[fn.ops[0]][d0])
    print(simulation.per_op_costs[fn.ops[1]][d0])


if __name__ == "__main__":
    test_per_op_costs()
