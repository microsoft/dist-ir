import argparse
from collections import defaultdict
import numpy as np
import onnx
from pathlib import Path
from contextlib import redirect_stdout

import dist_ir
from dist_ir.importer import import_from_onnx, parse_tensor_from_file
from dist_ir.ir import cpprint, pformat, Topology, Value
from dist_ir.executor import infer_types, SequentialExecutor, Simulator
from dist_ir.executor.cost_model import CostModel
from dist_ir.ir.type import Bool, Float, Int64, Tensor

ONNX_MODEL_PATH = "onnx_models"
BERT_TRAINING_DATA_PATH = "bert_training_data"
NUM_FEATURES_PER_SAMPLE = 6
model = "bert_tiny_bw.onnx"

np.random.seed(0)


def subfinder_dp(mylist, pattern):
    """Finds the maximum number of non-overlapping appearances of a sublist."""
    list_len = len(mylist)
    pattern_len = len(pattern)
    counts = [0] * list_len
    for i in range(list_len - 1, -1, -1):
        if mylist[i] == pattern[0] and mylist[i : i + pattern_len] == pattern:
            counts[i] = max(
                1 + (0 if pattern_len + i >= list_len else counts[i + pattern_len]),
                0 if i == list_len - 1 else counts[i + 1],
            )
        elif i < list_len - 1:
            counts[i] = counts[i + 1]
    return counts[0]


def subfinder(mylist, pattern):
    # https://stackoverflow.com/a/12576755
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i : i + len(pattern)] == pattern:
            matches.append(pattern)
    return matches


def get_stages_from_types(ops, op_types, stage_types, op_to_stage_map):
    """Given a list of op types organized into stages, returns the correpsonding op stages."""
    i = 0
    stages = []
    while i < len(op_types):
        if tuple(op_types[i : i + len(stage_types)]) == stage_types:
            stage = ops[i : i + len(stage_types)]
            for op in stage:
                if op in op_to_stage_map:
                    return []
            else:
                stages.append(tuple(stage))
                i += len(stage_types)
        else:
            i += 1
    return stages


def get_stages(function, num_attention_blocks=3):
    """Partitions the model by identifying the largest repeating subfunctions."""
    stage_counts = {}
    op_types = [op.op_type for op in function.ops]

    # Find the largest non-overlapping lists of op types that repeat
    # once for every attention block.
    for i in range(len(op_types)):
        for j in range(i + 1, len(op_types)):
            stage_types = op_types[i:j]
            if tuple(stage_types) in stage_counts:
                continue
            count = subfinder_dp(op_types, stage_types)
            stage_counts[tuple(stage_types)] = count
    results = sorted(
        [k for k, v in stage_counts.items() if v == num_attention_blocks],
        key=lambda x: len(x),
        reverse=True,
    )

    # Convert the lists of op types into stages of ops.
    op_to_stage_map = {}
    for stage_types in results:
        stages = get_stages_from_types(
            function.ops, op_types, stage_types, op_to_stage_map
        )
        for stage in stages:
            for op in stage:
                op_to_stage_map[op] = stage
    stages = set(op_to_stage_map.values())

    # Convert the list of op stages to a list of subfunctions.
    ordered_stages = []
    current_stage = []
    for op in function.ops:
        if op not in op_to_stage_map:
            current_stage.append(op)
        else:
            if len(current_stage) > 0:
                ordered_stages.append(tuple(current_stage))
                current_stage = []
            stage = op_to_stage_map[op]
            if stage not in ordered_stages:
                ordered_stages.append(stage)
    if len(current_stage) > 0:
        ordered_stages.append(tuple(current_stage))
        current_stage = []
    subfunctions = [
        function.get_subfunction(stage, name=f"stage{i}")
        for i, stage in enumerate(ordered_stages)
    ]

    # Verify that the list of subfunctions spans the full function in the correct order.
    i = 0
    for subfunction in subfunctions:
        assert function.ops[i : i + len(subfunction.ops)] == subfunction.ops
        i += len(subfunction.ops)
    assert i == len(function.ops)

    return subfunctions


def main(args):
    batch_size = 64
    max_seq_len = 512
    max_pred_per_seq = 20
    training_mode = True
    learning_rate = 1.0e-2
    onnx_model_path = Path(__file__).parent / ONNX_MODEL_PATH / model
    topology = Topology()
    d = topology.add_device("gpu")
    device_speeds = {"gpu": 1.0e13}
    function, input_data = import_from_onnx(
        onnx_model_path, default_device=d, parse_input_data=args.parse_input_data
    )
    # TODO: Cache function and input data
    cpprint(function)

    # Read input data from file and create minibatch.
    samples = defaultdict(list)
    for i in range(batch_size):
        sample_input_paths = [
            Path(__file__).parent
            / BERT_TRAINING_DATA_PATH
            / f"sample{i}"
            / f"feature{j}.pb"
            for j in range(NUM_FEATURES_PER_SAMPLE)
        ]
        for j in range(NUM_FEATURES_PER_SAMPLE):
            samples[j].append(parse_tensor_from_file(sample_input_paths[j]))
    inputs = [np.stack(samples[j], axis=0) for j in range(NUM_FEATURES_PER_SAMPLE)]
    input_value_order = [0, 2, 4, 3, 5, 1]
    for i, inp in enumerate(inputs):
        j = input_value_order[i]
        input_data[function.inputs[j]] = inp
        print(function.inputs[j].name, inp.shape)
    input_types = [
        Tensor(dtype=Int64(), shape=tuple(inp.shape), device=d) for inp in inputs
    ]
    input_types += [Float(), Bool()]
    input_types += [inp.type for inp in function.inputs[len(input_types) :]]
    assert function.inputs[6].name == "Learning_Rate"
    assert function.inputs[7].name == "training_mode"
    input_data[function.inputs[6]] = learning_rate
    input_data[function.inputs[7]] = training_mode

    # Partition the model into the largest possible repeating subfunctions.
    if args.get_stages:
        stages = get_stages(function)
        for stage in stages:
            cpprint(stage)
            print()

    # Simulate the model.
    if args.simulate:
        input_values = [
            Value(function.inputs[i].name, input_types[i]) for i in range(8)
        ] + list(function.inputs[8:])
        function = infer_types(function, input_values)
        cpprint(function)

        simulator = Simulator(CostModel(topology, device_speeds))
        state = simulator.interpret(function, input_types)

    # Run the model with real data.
    if args.run_sequential_executor:
        ex = SequentialExecutor("numpy")
        outputs = ex.compute(function, [input_data[i] for i in function.inputs])


def get_missing_op_types():
    onnx_model_path = Path(__file__).parent / ONNX_MODEL_PATH / model
    onnx_model = onnx.load(onnx_model_path)
    missing_op_types = set()
    for node in onnx_model.graph.node:
        if node.op_type not in dist_ir.ir.op_register.OpRegister:
            missing_op_types.add(node.op_type)
    for op_type in missing_op_types:
        print(op_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BERT-Tiny example")
    parser.add_argument(
        "--parse_input_data",
        action="store_true",
        default=False,
        help="Parse input data from ONNX file",
    )
    parser.add_argument(
        "--get_stages",
        action="store_true",
        default=False,
        help="Partition the model into logical stages",
    )
    parser.add_argument(
        "--run_sequential_executor",
        action="store_true",
        default=False,
        help="Run the model with real data",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Simulate the model",
    )
    # get_missing_op_types()
    args = parser.parse_args()
    main(args)
