from dist_ir.ir import Device, Module
from dist_ir.ir.type import Float, Tensor
from dist_ir.scheduler import FIFOScheduler, PipeDreamScheduler


def _construct_module_and_partition_map():
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

    partition_map = {
        "MatMul0": d0,
        "MatMul1": d1,
        "Loss": d1,
        "LossGrad": d1,
        "MatMul1Grad": d1,
        "MatMul0Grad": d0,
    }

    return (module, partition_map)


def test_fifo_scheduler():
    (module, partition_map) = _construct_module_and_partition_map()
    (d0, d1) = sorted(set(partition_map.values()))
    scheduler = FIFOScheduler(num_microbatches=2)
    schedule = scheduler.schedule(module, partition_map)

    ref_schedule = [
        {d0: ("MatMul0", 0)},
        {d0: ("MatMul0", 1), d1: ("MatMul1", 0)},
        {d1: ("Loss", 0)},
        {d1: ("LossGrad", 0)},
        {d1: ("MatMul1Grad", 0)},
        {d0: ("MatMul0Grad", 0), d1: ("MatMul1", 1)},
        {d1: ("Loss", 1)},
        {d1: ("LossGrad", 1)},
        {d1: ("MatMul1Grad", 1)},
        {d0: ("MatMul0Grad", 1)},
    ]

    assert schedule == ref_schedule


def test_pipedream_scheduler():
    (module, partition_map) = _construct_module_and_partition_map()
    (d0, d1) = sorted(set(partition_map.values()))
    scheduler = PipeDreamScheduler(num_microbatches=2)
    schedule = scheduler.schedule(module, partition_map)

    ref_schedule = [
        {d0: ("MatMul0", 0)},
        {d0: ("MatMul0", 1), d1: ("MatMul1", 0)},
        {d1: ("LossGrad", 0)},
        {d1: ("Loss", 0)},
        {d1: ("MatMul1Grad", 0)},
        {d0: ("MatMul0Grad", 0), d1: ("MatMul1", 1)},
        {d1: ("LossGrad", 1)},
        {d1: ("Loss", 1)},
        {d1: ("MatMul1Grad", 1)},
        {d0: ("MatMul0Grad", 1)},
    ]

    assert schedule == ref_schedule
