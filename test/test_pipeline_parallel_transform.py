from collections import OrderedDict
import numpy as np

from dist_ir.executor import SequentialExecutor, infer_types
from dist_ir.importer import import_from_onnx
from dist_ir.ir import cpprint
from dist_ir.ir.device import Device
from dist_ir.transforms import PipelineParallelTransform, PipeDreamScheduler

from . import pipeline_parallel_utils as utils
from examples.mlp import mlp


def test_mnist_fw_bw():
    (function, partition_map) = utils.construct_function_and_partition_map()
    # partition_map: subfunction -> device
    # TODO how about partition_map: Op -> stage?
    (d0, d1) = sorted(set(partition_map.values()))
    stages = list(partition_map.keys())
    # TODO can't schedule be inferred from partition_map?
    # schedule: List[Device -> (subfunction, microbatch_id)]
    schedule = [
        {d0: (stages[0], 0)},
        {d0: (stages[0], 1), d1: (stages[1], 0)},
        {d1: (stages[2], 0)},
        {d0: (stages[3], 0), d1: (stages[1], 1)},
        {d1: (stages[2], 1)},
        {d0: (stages[3], 1)},
    ]
    transform = PipelineParallelTransform(
        num_microbatches=2,
        batch_dims={function.inputs[0]: 0, function.inputs[1]: 0},
        reduction_params={
            function.outputs[0]: {"op_type": "Concat", "dim": 0},  # l
            function.outputs[1]: {"op_type": "Add"},  # dwB
            function.outputs[2]: {"op_type": "Concat", "dim": 0},  # dx
            function.outputs[3]: {"op_type": "Add"},  # dwA
        },
        partition_map=partition_map,
        schedule=schedule,
    )
    transformed_function = transform.apply(function)

    print("-" * 88)
    print("Original function")
    print("-" * 88)
    cpprint(function)
    print()
    print("-" * 88)
    print("Transformed function")
    print("-" * 88)
    cpprint(transformed_function)

    batch_size = 16
    ex = SequentialExecutor("numpy")
    _x = np.arange(batch_size * 4).reshape((batch_size, 4))
    _z = np.ones((batch_size, 1))
    _wA = np.ones((4, 2))
    _wB = np.ones((2, 1))
    orig_res = ex.compute(function, [_x, _z, _wA, _wB])

    transformed_res = ex.compute(transformed_function, [_x, _z, _wA, _wB])

    print("-" * 88)
    print("Original function results")
    print("-" * 88)
    print(orig_res)
    print()
    print("-" * 88)
    print("Transformed function results")
    print("-" * 88)
    print(transformed_res)
    print()

    for a, b in zip(orig_res, transformed_res):
        np.testing.assert_array_almost_equal(a, b)


def fusion_pp_expt():
    # Given: function and device mapping: Op -> Device
    batch_size = 8
    hidden_dim = 6
    num_layers = 2

    d0 = Device(0, "cpu")
    seq_mlp = mlp(batch_size, hidden_dim, hidden_dim, hidden_dim, num_layers, d0)
    cpprint(seq_mlp)

    device_mapping = {  # devices are 0-indexed
        seq_mlp.ops[0]: 0,
        seq_mlp.ops[1]: 0,
        seq_mlp.ops[2]: 1,
        seq_mlp.ops[3]: 1,
        seq_mlp.ops[4]: 1,
        seq_mlp.ops[5]: 1,
        seq_mlp.ops[6]: 1,
        seq_mlp.ops[7]: 1,
        seq_mlp.ops[8]: 0,
        seq_mlp.ops[9]: 0,
    }

    # PP transform it
    def create_partition_map(devices, device_mapping):
        """Walks through fn.ops and identifies stages by finding consecutive
        ranges of ops mapped to the same device.

        devices: list of Devices
        device_mapping: Op -> device ID (index in devices)

        Returns: map from stages (Functions) to Device.
        """
        partition_map = OrderedDict()
        last_index = 0
        last_device = device_mapping[seq_mlp.ops[0]]
        for i in range(1, len(seq_mlp.ops) + 1):
            if i == len(seq_mlp.ops) or device_mapping[seq_mlp.ops[i]] != last_device:
                # We have identified a new stage: [last_index, i)
                partition_map[
                    seq_mlp.get_subfunction(seq_mlp.ops[last_index:i], name=f"foo_{i}")
                ] = devices[last_device]
                last_index = i
                if i < len(seq_mlp.ops):
                    last_device = device_mapping[seq_mlp.ops[i]]
        return partition_map

    devices = [Device(1, "gpu"), Device(2, "gpu")]
    partition_map = create_partition_map(devices, device_mapping)
    schedule = PipeDreamScheduler(num_microbatches=2).schedule(seq_mlp, partition_map)

    transform = PipelineParallelTransform(
        num_microbatches=2,
        batch_dims={seq_mlp.inputs[0]: 0, seq_mlp.inputs[1]: 0},
        reduction_params={
            seq_mlp.outputs[0]: {"op_type": "Concat", "dim": 0},  # l
            seq_mlp.outputs[1]: {"op_type": "Add"},  # dwB
            seq_mlp.outputs[2]: {"op_type": "Concat", "dim": 0},  # dx
            seq_mlp.outputs[3]: {"op_type": "Add"},  # dwA
        },
        partition_map=partition_map,
        schedule=schedule,
    )
    pp_mlp = transform.apply(seq_mlp)
    pp_mlp = infer_types(pp_mlp, pp_mlp.inputs)
    cpprint(pp_mlp)

    # Run simulator and get timings


def bert_base_pp_expt():
    onnx_model_path = "/home/t-sikris/bert_data/graphs/bert-base-uncased_L_12_H_768_A_12_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12_shallow.onnx"
    fn, _ = import_from_onnx(onnx_model_path, parse_input_data=False)
    cpprint(fn)


if __name__ == "__main__":
    # fusion_pp_expt()
    # test_mnist_fw_bw()
    bert_base_pp_expt()
