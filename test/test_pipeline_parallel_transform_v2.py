from dist_ir.ir import FunctionMaker, Topology, cpprint
from dist_ir.ir.type import Float, Tensor
from dist_ir.executor import infer_types
from dist_ir.transforms import pipeline_parallel_transform_v2
from examples.mlp import mlp


def _get_op_to_stage_map(function, num_hidden_layers, num_devices):
    op_to_stage_map = {}
    num_forward_ops = 2 * num_hidden_layers

    for i, op in enumerate(function.ops):
        if i < num_forward_ops:
            stage = int(i // (num_forward_ops // num_devices))
        elif i >= num_forward_ops and i < num_forward_ops + 2:
            stage = num_devices - 1
        else:
            stage = int(
                num_devices
                - 1
                - (i - 2 - num_forward_ops) // (num_forward_ops // num_devices)
            )
        assert stage <= num_devices - 1
        op_to_stage_map[op] = stage
    return op_to_stage_map


def test_pipeline_parallel_transform_v2():
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
    function = infer_types(function, function.inputs)
    op_to_stage_map = _get_op_to_stage_map(function, num_hidden_layers, num_devices)
    partitioned_device_map = {
        topology.devices[0]: {0: topology.devices[1], 1: topology.devices[2]},
    }

    transformed_function = pipeline_parallel_transform_v2(
        function,
        op_to_stage_map,
        partitioned_device_map,
        batch_inputs=set(function.inputs[:2]),
        reduction_params={
            output: {"op_type": "Add"}
            if "dw" in output.name
            else {"op_type": "Concat", "dim": 0}
            for output in function.outputs
        },
        num_microbatches=2,
    )
    transformed_function = infer_types(
        transformed_function, transformed_function.inputs
    )
    cpprint(transformed_function)


if __name__ == "__main__":
    test_pipeline_parallel_transform_v2()
