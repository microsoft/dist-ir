from dist_ir.transforms import FIFOScheduler, PipeDreamScheduler
import pipeline_parallel_utils as utils


def test_fifo_scheduler():
    (module, partition_map) = utils.construct_module_and_partition_map()
    (d0, d1) = sorted(set(partition_map.values()))
    scheduler = FIFOScheduler(num_microbatches=2)
    schedule = scheduler.schedule(module, partition_map)

    stages = list(partition_map.keys())
    ref_schedule = [
        {d0: (stages[0], 0)},
        {d0: (stages[0], 1), d1: (stages[1], 0)},
        {d1: (stages[1], 1)},
        {d1: (stages[2], 0)},
        {d0: (stages[3], 0), d1: (stages[2], 1)},
        {d0: (stages[3], 1)},
    ]

    assert schedule == ref_schedule


def test_pipedream_scheduler():
    (module, partition_map) = utils.construct_module_and_partition_map()
    (d0, d1) = sorted(set(partition_map.values()))
    scheduler = PipeDreamScheduler(num_microbatches=2)
    schedule = scheduler.schedule(module, partition_map)

    stages = list(partition_map.keys())
    ref_schedule = [
        {d0: (stages[0], 0)},
        {d0: (stages[0], 1), d1: (stages[1], 0)},
        {d1: (stages[2], 0)},
        {d0: (stages[3], 0), d1: (stages[1], 1)},
        {d1: (stages[2], 1)},
        {d0: (stages[3], 1)},
    ]

    assert schedule == ref_schedule
