# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import OrderedDict

from dist_ir.ir import Device, FunctionMaker
from dist_ir.ir.type import Int32, Float32, Tensor


def construct_function_and_partition_map():
    function = FunctionMaker()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")
    batch_size = 16
    x = function.add_input_value(
        "x", Tensor(dtype=Float32(), shape=(batch_size, 4), device=d0)
    )
    z = function.add_input_value(
        "z", Tensor(dtype=Float32(), shape=(batch_size, 1), device=d0)
    )
    n = function.add_input_value("n", Int32(device=d0))
    wA = function.add_input_value(
        "wA", Tensor(dtype=Float32(), shape=(4, 2), device=d0)
    )
    wB = function.add_input_value(
        "wB", Tensor(dtype=Float32(), shape=(2, 1), device=d0)
    )
    a = function.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = function.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    l = function.add_op("Loss", "Loss", inputs=[y, z, n], output_names=["l"])
    dl = function.add_op(
        "LossGrad",
        "LossGrad",
        inputs=[y, z, n],
        output_names=["dl"],
    )
    da, dwB = function.add_op(
        "MatMulGrad", "MatMul1Grad", inputs=[a, wB, dl], output_names=["da", "dwB"]
    )
    _, dwA = function.add_op(
        "MatMulGrad", "MatMul0Grad", inputs=[x, wA, da], output_names=["dx", "dwA"]
    )
    function = function.finalize()

    stages = [
        function.get_subfunction([function.ops[0]], name="f0"),
        function.get_subfunction(function.ops[1:3], name="f1"),
        function.get_subfunction(function.ops[3:5], name="b1"),
        function.get_subfunction([function.ops[5]], name="b0"),
    ]

    partition_map = OrderedDict(
        [(stages[0], d0), (stages[1], d1), (stages[2], d1), (stages[3], d0)]
    )

    return (function, partition_map)
