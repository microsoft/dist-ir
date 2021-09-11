import itertools
import numpy as np
from pathlib import Path
import pytest
import torch

from dist_ir.executor import SequentialExecutor, ConcreteValue
from dist_ir.ir import cpprint
from examples.gpt2 import get_transformed_function_and_input_data, run_pytorch
from dist_ir.utils import constants

# Assume the onnx file is stored in the repository root
MODEL_PATH = (Path(__file__).parent.parent / "gpt2-10.onnx").absolute()

np.random.seed(42)


def _run_gpt(
    device_throughput=constants.DEFAULT_DEVICE_THROUGHPUT,
    dram_bandwidth=constants.DEFAULT_DRAM_BANDWIDTH,
    kernel_launch_overhead=constants.DEFAULT_KERNEL_LAUNCH_OVERHEAD,
    network_bandwidth=constants.DEFAULT_NETWORK_BANDWIDTH,
    batch_size=256,
    dp_degree=1,
    hp_degree=1,
    pp_degree=1,
    num_microbatches=1,
    n_layer=4,
    n_head=12,
    n_embd=768,
    use_real_weights=True,
    use_pytorch_backend=False,
    debug_stacktrace=False,
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
        kernel_launch_overhead,
        network_bandwidth,
        batch_size,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
        n_layer,
        n_head,
        n_embd,
        use_real_weights=use_real_weights,
    )
    if verbose:
        cpprint(transformed_function)
    if use_real_weights:
        if use_pytorch_backend:
            world_size = dp_degree * hp_degree * pp_degree
            outputs, _ = run_pytorch(
                transformed_function,
                initialized_input_data,
                world_size,
                use_gpu=torch.cuda.device_count() >= world_size,
                debug_stacktrace=debug_stacktrace,
            )
            outputs = tuple(
                ConcreteValue(v.numpy(), None if t.type is None else t.type.device)
                for v, t in zip(
                    tuple(itertools.chain.from_iterable(outputs)),
                    transformed_function.outputs,
                )
            )
        else:
            ex = SequentialExecutor("numpy")
            outputs = ex.compute(transformed_function, initialized_input_data)
        return outputs


def _test(
    original_outputs,
    dp_degree=1,
    hp_degree=1,
    pp_degree=1,
    num_microbatches=1,
    use_pytorch_backend=False,
    debug_stacktrace=False,
):

    # Test with real weights
    transformed_outputs = _run_gpt(
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=num_microbatches,
        use_pytorch_backend=use_pytorch_backend,
        debug_stacktrace=debug_stacktrace,
    )
    assert len(transformed_outputs) == dp_degree * hp_degree
    for i in range(len(transformed_outputs)):
        np.testing.assert_array_almost_equal(
            original_outputs[0].val, transformed_outputs[i].val, decimal=2
        )


@pytest.fixture(scope="session")
def original_outputs():
    return _run_gpt()


@pytest.mark.parametrize(
    ("dp_degree", "hp_degree", "pp_degree"),
    list(itertools.product([1, 2], [1, 2], [1, 2])),
)
def test_reference_execution(original_outputs, dp_degree, hp_degree, pp_degree):
    _test(
        original_outputs,
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=pp_degree,
    )


@pytest.mark.parametrize(
    ("dp_degree", "hp_degree", "pp_degree"),
    [
        x
        for x in list(itertools.product([1, 2], [1, 2], [1, 2]))
        if (x[0] * x[1] * x[2]) <= torch.cuda.device_count()
    ],
)
def test_pytorch_backend(
    original_outputs, dp_degree, hp_degree, pp_degree, debug_stacktrace=False
):
    _test(
        original_outputs,
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=pp_degree,
        use_pytorch_backend=True,
        debug_stacktrace=debug_stacktrace,
    )


@pytest.mark.parametrize(
    ("dp_degree", "hp_degree", "pp_degree"),
    list(itertools.product([1, 2], [1, 2], [1, 2])),
)
def test_mixed_simulation(dp_degree, hp_degree, pp_degree):
    _run_gpt(
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=pp_degree,
        use_real_weights=False,
    )
