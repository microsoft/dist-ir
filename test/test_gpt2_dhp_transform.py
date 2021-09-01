import itertools
import numpy as np
from pathlib import Path
import pytest

from dist_ir.executor import SequentialExecutor, ConcreteValue
from dist_ir.ir import cpprint
from examples.gpt2 import get_transformed_function_and_input_data, run_pytorch

# Assume the onnx file is stored in the repository root
MODEL_PATH = (Path(__file__).parent.parent / "gpt2-10.onnx").absolute()

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
    n_layer=4,
    n_head=12,
    n_embd=768,
    use_real_weights=True,
    use_pytorch_backend=False,
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
        use_real_weights=use_real_weights,
    )
    if verbose:
        cpprint(transformed_function)
    if use_real_weights:
        if use_pytorch_backend:
            world_size = dp_degree * hp_degree * pp_degree
            outputs, _ = run_pytorch(
                transformed_function, initialized_input_data, world_size, use_gpu=False
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
):

    # Test with real weights
    transformed_outputs = _run_gpt(
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=num_microbatches,
        use_pytorch_backend=use_pytorch_backend,
    )
    assert len(transformed_outputs) == dp_degree * hp_degree
    for i in range(len(transformed_outputs)):
        np.testing.assert_array_almost_equal(
            original_outputs[0].val, transformed_outputs[i].val, decimal=2
        )

    # Test with mixed implementations
    # TODO: Factor this out into a separate test?
    _run_gpt(
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=num_microbatches,
        use_real_weights=False,
    )


@pytest.fixture(scope="session")
def original_outputs():
    return _run_gpt()


@pytest.mark.parametrize("dp_degree", [2, 4])
def test_dp_only(original_outputs, dp_degree):
    _test(original_outputs, dp_degree=dp_degree)


@pytest.mark.parametrize("hp_degree", [2, 4])
def test_hp_only(original_outputs, hp_degree):
    _test(original_outputs, hp_degree=hp_degree)


@pytest.mark.parametrize(
    ("pp_degree", "num_microbatches"), list(itertools.product([2, 4], [2, 4]))
)
def test_pp_only(original_outputs, pp_degree, num_microbatches):
    _test(original_outputs, pp_degree=pp_degree, num_microbatches=num_microbatches)


@pytest.mark.parametrize(
    ("dp_degree", "hp_degree"),
    list(itertools.product([2, 4], [2, 4])),
)
def test_dp_hp(original_outputs, dp_degree, hp_degree):
    _test(original_outputs, dp_degree=dp_degree, hp_degree=hp_degree)


@pytest.mark.parametrize(
    ("dp_degree", "pp_degree"),
    list(itertools.product([2, 4], [2, 4])),
)
def test_dp_pp(original_outputs, dp_degree, pp_degree):
    _test(
        original_outputs, dp_degree=dp_degree, pp_degree=pp_degree, num_microbatches=2
    )


@pytest.mark.parametrize(
    ("hp_degree", "pp_degree"),
    list(itertools.product([2, 4], [2, 4])),
)
def test_hp_pp(original_outputs, hp_degree, pp_degree):
    _test(
        original_outputs, hp_degree=hp_degree, pp_degree=pp_degree, num_microbatches=2
    )


@pytest.mark.parametrize(
    ("dp_degree", "hp_degree", "pp_degree"),
    list(itertools.product([2], [2], [2])),
)
def test_dp_hp_pp(original_outputs, dp_degree, hp_degree, pp_degree):
    _test(
        original_outputs,
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=2,
    )


@pytest.mark.parametrize(
    ("dp_degree", "hp_degree", "pp_degree"),
    list(itertools.product([1, 2], [1, 2], [1, 2])),
)
def test_pytorch_backend(original_outputs, dp_degree, hp_degree, pp_degree):
    _test(
        original_outputs,
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=pp_degree,
        use_pytorch_backend=True,
    )
