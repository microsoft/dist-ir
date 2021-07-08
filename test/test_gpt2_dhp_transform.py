from collections import defaultdict, OrderedDict
import itertools
import numpy as np
import pytest
import re

from dist_ir.executor import SequentialExecutor
from dist_ir.ir import cpprint
from examples.gpt2 import get_transformed_function_and_input_data

# TODO: Make this configurable?
MODEL_PATH = "/lfs/1/keshav2/gpt2/gpt2-10.onnx"

np.random.seed(42)


def _run_gpt(
    device_throughput=1.4e13,
    dram_bandwidth=9e11,
    network_bandwidth=64,
    batch_size=256,
    dp_degree=1,
    hp_degree=1,
    pp_degree=1,
    num_microbatches=1,
    n_layer=12,
    n_head=12,
    n_embd=768,
    verbose=False,
):
    (
        transformed_function,
        initialized_input_data,
        topology,
    ) = get_transformed_function_and_input_data(
        MODEL_PATH,
        device_throughput,
        dram_bandwidth,
        network_bandwidth,
        batch_size,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
        n_layer,
        n_head,
        n_embd,
    )
    if verbose:
        cpprint(transformed_function)
    ex = SequentialExecutor("numpy")
    outputs = ex.compute(transformed_function, initialized_input_data)
    return outputs


def _test(dp_degree=1, hp_degree=1, pp_degree=1, num_microbatches=1):
    # TODO: Figure out how to cache the downloaded model from 
    # https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/model/gpt2-10.onnx?raw=true
    return
    original_outputs = _run_gpt()
    transformed_outputs = _run_gpt(
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=num_microbatches,
    )
    assert len(transformed_outputs) == dp_degree * hp_degree
    for i in range(len(transformed_outputs)):
        np.testing.assert_array_almost_equal(
            original_outputs[0], transformed_outputs[i], decimal=2
        )


@pytest.mark.parametrize("dp_degree", [2, 4, 8])
def test_dp_only(dp_degree):
    _test(dp_degree=dp_degree)


@pytest.mark.parametrize("hp_degree", [2, 4, 8])
def test_hp_only(hp_degree):
    _test(hp_degree=hp_degree)


@pytest.mark.parametrize(
    ("pp_degree", "num_microbatches"), list(itertools.product([2, 4, 8], [2, 4, 8]))
)
def test_pp_only(pp_degree, num_microbatches):
    _test(pp_degree=pp_degree, num_microbathces=num_microbatches)


@pytest.mark.parametrize(
    ("dp_degree", "hp_degree"),
    list(itertools.product([2, 4, 8], [2, 4, 8])),
)
def test_dp_hp(dp_degree, hp_degree):
    _test(dp_degree=dp_degree, hp_degree=hp_degree)


@pytest.mark.parametrize(
    ("dp_degree", "pp_degree"),
    list(itertools.product([2, 4, 8], [2, 4, 8])),
)
def test_dp_pp(dp_degree, pp_degree):
    _test(dp_degree=dp_degree, pp_degree=pp_degree, num_microbatches=2)


@pytest.mark.parametrize(
    ("hp_degree", "pp_degree"),
    list(itertools.product([2, 4, 8], [2, 4, 8])),
)
def test_hp_pp(hp_degree, pp_degree):
    _test(hp_degree=hp_degree, pp_degree=pp_degree, num_microbatches=2)


@pytest.mark.parametrize(
    ("dp_degree", "hp_degree", "pp_degree"),
    list(itertools.product([2, 4], [2, 4], [2, 4])),
)
def test_dp_hp_pp(dp_degree, hp_degree, pp_degree):
    _test(
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=2,
    )
