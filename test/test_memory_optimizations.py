# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from dist_ir.ir import FunctionMaker, get_uniform_topology, Value
from dist_ir.ir.type import Int32, Float16, Tensor
from dist_ir.executor import (
    CostModel,
    Simulator,
)


def gradient_checkpointing(topology, optimize):
    d = topology.devices[0]
    fn = FunctionMaker()
    n = fn.add_input_value("n", Int32(device=d))
    w1 = fn.add_input_value("w1", Tensor(shape=(512, 512), device=d, dtype=Float16()))
    w2 = fn.add_input_value("w2", Tensor(shape=(512, 512), device=d, dtype=Float16()))
    x = fn.add_input_value("x", Tensor(shape=(512, 512), device=d, dtype=Float16()))
    y = fn.add_input_value("y", Tensor(shape=(512, 512), device=d, dtype=Float16()))
    as_f = fn.add_op(op_type="MatMul", inputs=[x, w1], output_names=["as_f"])
    p_1 = fn.add_op(op_type="MatMul", inputs=[as_f, w2], output_names=["p_1"])
    dp = fn.add_op(op_type="LossGrad", inputs=[p_1, y, n], output_names=["dp"])
    if optimize:
        as_b = fn.add_op(op_type="MatMul", inputs=[x, w1], output_names=["as_b"])
    else:
        as_b = as_f
    das, dw2 = fn.add_op(
        op_type="MatMulGrad", inputs=[as_b, w2, dp], output_names=["das", "dw2"]
    )
    _, dw1 = fn.add_op(
        op_type="MatMulGrad", inputs=[x, w1, das], output_names=["dx", "dw1"]
    )
    fn.add_op(
        op_type="SGDOptimizer",
        inputs=([w1, w2, dw1, dw2]),
        attributes={"lr": 0},  # Set learning rate to 0 to prevent precision issues
        output_names=["w1", "w2"],
    )

    return fn.finalize()


def zero(topology, optimize):
    d1 = topology.devices[0]
    d2 = topology.devices[1]

    fn = FunctionMaker()
    n_1 = fn.add_input_value("n_1", Int32(device=d1))
    n_2 = fn.add_input_value("n_1", Int32(device=d2))
    w1 = fn.add_input_value("w1", Tensor(shape=(512, 512), device=d1, dtype=Float16()))
    w2 = fn.add_input_value("w2", Tensor(shape=(512, 512), device=d2, dtype=Float16()))
    x_1 = fn.add_input_value("x", Tensor(shape=(256, 512), device=d1, dtype=Float16()))
    x_2 = fn.add_input_value("x", Tensor(shape=(256, 512), device=d2, dtype=Float16()))
    y_1 = fn.add_input_value("y", Tensor(shape=(256, 512), device=d1, dtype=Float16()))
    y_2 = fn.add_input_value("y", Tensor(shape=(256, 512), device=d2, dtype=Float16()))

    if not optimize:
        w1_2_f = fn.add_input_value(
            "w1_2_f", Tensor(shape=(512, 512), device=d2, dtype=Float16())
        )
        w1_2_b = w1_2_f
        w2_1_f = fn.add_input_value(
            "w2_1_f", Tensor(shape=(512, 512), device=d1, dtype=Float16())
        )
        w2_1_b = w2_1_f

    if optimize:
        w1_2_f = fn.add_op(
            op_type="Send",
            inputs=[w1],
            attributes={"device": d2},
            output_names=["w1_2_f"],
        )
    as_1 = fn.add_op(op_type="MatMul", inputs=[x_1, w1], output_names=["as_1"])
    as_2 = fn.add_op(op_type="MatMul", inputs=[x_2, w1_2_f], output_names=["as_2"])
    if optimize:
        w2_1_f = fn.add_op(
            op_type="Send",
            inputs=[w2],
            attributes={"device": d1},
            output_names=["w2_1_f"],
        )
    p_1 = fn.add_op(op_type="MatMul", inputs=[as_1, w2_1_f], output_names=["p_1"])
    p_2 = fn.add_op(op_type="MatMul", inputs=[as_2, w2], output_names=["p_2"])
    dp_1 = fn.add_op(op_type="LossGrad", inputs=[p_1, y_1, n_1], output_names=["dp_1"])
    dp_2 = fn.add_op(op_type="LossGrad", inputs=[p_2, y_2, n_2], output_names=["dp_2"])
    if optimize:
        w2_1_b = fn.add_op(
            op_type="Send",
            inputs=[w2],
            attributes={"device": d1},
            output_names=["w2_1_b"],
        )
    das_1, dw2_1 = fn.add_op(
        op_type="MatMulGrad",
        inputs=[as_1, w2_1_b, dp_1],
        output_names=["das_1", "dw2_1"],
    )
    das_2, dw2_2 = fn.add_op(
        op_type="MatMulGrad", inputs=[as_2, w2, dp_2], output_names=["das_2", "dw2_2"]
    )
    _, dw1_1 = fn.add_op(
        op_type="MatMulGrad", inputs=[x_1, w1, das_1], output_names=["dx_1", "dw1_1"]
    )
    if optimize:
        w1_2_b = fn.add_op(
            op_type="Send",
            inputs=[w1],
            attributes={"device": d2},
            output_names=["w1_2_b"],
        )
    _, dw1_2 = fn.add_op(
        op_type="MatMulGrad",
        inputs=[x_2, w1_2_b, das_2],
        output_names=[
            "dx_2",
            "dw1_2",
        ],
    )
    if optimize:
        dw1 = fn.add_op(
            "MPIReduce",
            inputs=[dw1_1, dw1_2],
            attributes={"device": d1},
            output_names=["dw1"],
        )
        dw2 = fn.add_op(
            "MPIReduce",
            inputs=[dw2_1, dw2_2],
            attributes={"device": d2},
            output_names=["dw2"],
        )
        fn.add_op(
            op_type="SGDOptimizer",
            inputs=([w1, w2, dw1, dw2]),
            attributes={"lr": 0},  # Set learning rate to 0 to prevent precision issues
            output_names=["w1", "w2"],
        )
    else:
        dw1_1, dw1_2 = fn.add_op(
            "MPIAllreduce", inputs=[dw1_1, dw1_2], output_names=["dw1_1", "dw1_2"]
        )
        dw2_1, dw2_2 = fn.add_op(
            "MPIAllreduce", inputs=[dw2_1, dw2_2], output_names=["dw2_1", "dw2_2"]
        )
        fn.add_op(
            op_type="SGDOptimizer",
            inputs=([w1, w1_2_f, w2_1_f, w2, dw1_1, dw1_2, dw2_1, dw2_2]),
            attributes={"lr": 0},  # Set learning rate to 0 to prevent precision issues
            output_names=["w1", "w1_2_f", "w2_1_f", "w2"],
        )

    return fn.finalize()


@pytest.mark.parametrize(
    "model, world_size",
    [(gradient_checkpointing, 1), (zero, 2)],
)
def test_optimization(model, world_size):
    topology = get_uniform_topology(world_size)
    peak_memory = {}
    for optimize in [False, True]:
        fn = model(topology, optimize)
        input_types = [inp.type for inp in fn.inputs]
        simulator = Simulator(CostModel(topology, None))
        simulation = simulator.simulate(fn, input_types)
        peak_memory[optimize] = simulation.peak_memory
    for d in topology.devices:
        # TODO: Convert to < once we enhance backend memory management
        assert peak_memory[True][d] <= peak_memory[False][d]


if __name__ == "__main__":
    test_optimization(gradient_checkpointing, 1)
    test_optimization(zero, 2)
