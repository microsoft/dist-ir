from collections import OrderedDict

from dist_ir.ir import Device, Module
from dist_ir.ir.type import Float, Tensor


def construct_module_and_partition_map():
    module = Module()

    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")
    batch_size = 16
    x = module.add_input_value(
        "x", Tensor(dtype=Float(), shape=(batch_size, 4), device=d0)
    )
    z = module.add_input_value(
        "z", Tensor(dtype=Float(), shape=(batch_size, 1), device=d0)
    )
    wA = module.add_input_value("wA", Tensor(dtype=Float(), shape=(4, 2), device=d0))
    wB = module.add_input_value("wB", Tensor(dtype=Float(), shape=(2, 1), device=d0))
    a = module.add_op("MatMul", "MatMul0", inputs=[x, wA], output_names=["a"])
    y = module.add_op("MatMul", "MatMul1", inputs=[a, wB], output_names=["y"])
    l = module.add_op(
        "Loss", "Loss", inputs=[y, z], attributes={"N": batch_size}, output_names=["l"]
    )
    dl = module.add_op(
        "LossGrad",
        "LossGrad",
        inputs=[y, z],
        attributes={"N": batch_size},
        output_names=["dl"],
    )
    da, dwB = module.add_op(
        "MatMulGrad", "MatMul1Grad", inputs=[a, wB, dl], output_names=["da", "dwB"]
    )
    _, dwA = module.add_op(
        "MatMulGrad", "MatMul0Grad", inputs=[x, wA, da], output_names=["dx", "dwA"]
    )
    module.set_outputs([l, dwA, dwB])
    module.finalize()

    stages = [
        module.get_submodule(("MatMul0",), name="f0"),
        module.get_submodule(("MatMul1", "Loss"), name="f1"),
        module.get_submodule(("LossGrad", "MatMul1Grad"), name="b1"),
        module.get_submodule(("MatMul0Grad",), name="b0"),
    ]

    partition_map = OrderedDict(
        [(stages[0], d0), (stages[1], d1), (stages[2], d1), (stages[3], d0)]
    )

    return (module, partition_map)