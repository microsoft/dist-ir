# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
import numpy as np
from pathlib import Path
import pytest
import torch

from dist_ir.executor import sequentially_execute, ConcreteValue
from dist_ir.ir import cpprint
from dist_ir.ir.type import Float16, Float32
from dist_ir.utils import constants
from examples.gpt2 import get_transformed_function_and_input_data, simulate, run_pytorch

# Assume the onnx file is stored in the repository root
MODEL_PATH = (Path(__file__).parent.parent / "gpt2-10.onnx").absolute()

np.random.seed(42)


def _run_gpt(
    dtype="fp32",
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
    verbose=False,
):
    (
        transformed_function,
        initialized_input_data,
        topology,
    ) = get_transformed_function_and_input_data(
        MODEL_PATH,
        dtype,
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
            results = run_pytorch(
                transformed_function,
                initialized_input_data,
                world_size,
                num_warmup=1,
                num_repetitions=1,
                use_gpu=torch.cuda.device_count() >= world_size,
            )
            outputs = results.per_rank_outputs
            outputs = tuple(
                ConcreteValue(v.numpy(), None if t.type is None else t.type.device)
                for v, t in zip(
                    tuple(itertools.chain.from_iterable(outputs)),
                    transformed_function.outputs,
                )
            )
        else:
            outputs = sequentially_execute(transformed_function, initialized_input_data)
        return outputs
    else:
        return simulate(transformed_function, initialized_input_data, topology)


def _test(
    original_outputs,
    dtype,
    dp_degree=1,
    hp_degree=1,
    pp_degree=1,
    num_microbatches=1,
    use_pytorch_backend=False,
):

    # Test with real weights
    transformed_outputs = _run_gpt(
        dtype=dtype,
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=num_microbatches,
        use_pytorch_backend=use_pytorch_backend,
    )
    assert len(transformed_outputs) == dp_degree * hp_degree
    for i in range(len(transformed_outputs)):
        if dtype == "fp32":
            assert transformed_outputs[i].val.dtype == np.float32
        else:
            assert transformed_outputs[i].val.dtype == np.float16
        np.testing.assert_array_almost_equal(
            original_outputs[0].val,
            transformed_outputs[i].val,
            decimal=(2 if dtype == "fp32" else 1),
        )


@pytest.fixture(scope="session")
def original_outputs():
    if torch.cuda.is_available():
        return {
            "fp16": _run_gpt(dtype="fp16", use_pytorch_backend=True),
            "fp32": _run_gpt(dtype="fp32", use_pytorch_backend=True),
        }
    else:
        return {
            "fp32": _run_gpt(dtype="fp32", use_pytorch_backend=True),
        }


@pytest.mark.parametrize(
    ("dp_degree", "hp_degree", "pp_degree"),
    list(itertools.product([1, 2], [1, 2], [1, 2])),
)
def test_reference_execution(original_outputs, dp_degree, hp_degree, pp_degree):
    _test(
        original_outputs["fp32"],
        dtype="fp32",
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=pp_degree,
    )


@pytest.mark.parametrize(
    ("dtype", "dp_degree", "hp_degree", "pp_degree"),
    list(
        itertools.product(
            ["fp16", "fp32"] if torch.cuda.is_available() else ["fp32"],
            [1, 2],
            [1, 2],
            [1, 2],
        )
    ),
)
def test_pytorch_backend(original_outputs, dtype, dp_degree, hp_degree, pp_degree):
    world_size = dp_degree * hp_degree * pp_degree
    if dtype == "fp16" and world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs available")
    elif world_size >= 8:
        pytest.skip("TODO: Fix CPU gloo backend bug")
    _test(
        original_outputs[dtype],
        dtype,
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=pp_degree,
        use_pytorch_backend=True,
    )


@pytest.mark.parametrize(
    ("dtype", "dp_degree", "hp_degree", "pp_degree"),
    list(itertools.product(["fp16", "fp32"], [1, 2], [1, 2], [1, 2])),
)
def test_mixed_simulation(dtype, dp_degree, hp_degree, pp_degree):
    simulation = _run_gpt(
        dtype=dtype,
        dp_degree=dp_degree,
        hp_degree=hp_degree,
        pp_degree=pp_degree,
        num_microbatches=pp_degree,
        use_real_weights=False,
    )
    # TODO: Verify that output dtypes are correct
