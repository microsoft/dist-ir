from dist_ir.ir import FunctionMaker, Topology, cpprint
from dist_ir.ir.type import Float, Tensor
from dist_ir.executor import infer_types
from dist_ir.transforms import spmd_transform
from examples.mlp import mlp


def test_spmd_transform():
    batch_size = 128
    input_dim = 32
    num_hidden_layers = 8
    num_devices = 2
    topology = Topology()
    for i in range(num_devices + 1):
        topology.add_device("gpu")
    function = mlp(
        batch_size,
        input_dim,
        input_dim,
        input_dim,
        num_hidden_layers,
        topology.devices[0],
    )
    transformed_function = spmd_transform(
        function,
        input_dims={function.inputs[0]: 0, function.inputs[1]: 0},
        reduction_params={
            output: {"op_type": "MPIAllreduce"}
            if "dw" in output.name
            else {"op_type": "MPIAllgather", "dim": 0}
            for output in function.outputs
        },
        devices=topology.devices[1:],
    )
    transformed_function = infer_types(
        transformed_function, transformed_function.inputs
    )
    cpprint(transformed_function)


if __name__ == "__main__":
    test_spmd_transform()
