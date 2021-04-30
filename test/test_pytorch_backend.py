import numpy as np
import pytest
import torch

from dist_ir.backend.torch import run_multiprocesses
from dist_ir.executor import SequentialExecutor
from dist_ir.executor.rank_projector import project
from dist_ir.executor.type_inference import infer_types
from dist_ir.ir import Device, FunctionMaker, cpprint, Value
from dist_ir.ir.type import Float, Tensor
from dist_ir.transforms import mlp_dhp_transform
from examples.mlp import mlp_inference, mlp_inference_dp


def create_owt_model(num_devices, num_layers):
    assert num_layers % 2 == 0

    fn = FunctionMaker()

    # Inputs
    weights = {}
    xs = {}
    for l in range(num_layers):
        for d in range(1, num_devices + 1):
            weights[l, d] = fn.add_input_value(f"w{l}_{d}", None)
    for d in range(1, num_devices + 1):
        xs[d] = fn.add_input_value(f"x_{d}", None)

    # Data parallel conv blocks: (using MatMuls for now)
    hs = []
    for d in range(1, num_devices + 1):
        h = xs[d]
        for l in range(num_layers // 2):
            h = fn.add_op(
                "MatMul", inputs=[h, weights[l, d]], output_names=[f"h{l}_{d}"]
            )
        hs.append(h)

    # Allgather the activations
    as_names = [f"hh{num_layers//2-1}_{d}" for d in range(1, num_devices + 1)]
    hs = fn.add_op(
        "MPIAllgather",
        inputs=hs,
        output_names=as_names,
        attributes={"dim": 0, "world_size": num_devices},
    )

    # Model parallel fully-connected layers: (again, MatMuls for now)
    hs = hs
    for l in range(num_layers // 2, num_layers):
        h_is = []
        for d in range(1, num_devices + 1):
            h_is.append(
                fn.add_op(
                    "MatMul",
                    inputs=[hs[d - 1], weights[l, d]],
                    output_names=[f"h{l}_{d}"],
                )
            )
        if l == num_layers - 1:
            hs = h_is
        else:
            out_names = [f"hh{l}_{d}" for d in range(1, num_devices + 1)]
            hs = fn.add_op(
                "MPIAllgather",
                inputs=h_is,
                output_names=out_names,
                attributes={"dim": 1, "world_size": num_devices},
            )

    fn.set_outputs(hs)
    return fn.finalize()


@pytest.mark.parametrize(["num_devices", "num_layers"], [(2, 4)])
def test_owt(num_devices, num_layers):
    fn = create_owt_model(num_devices, num_layers)

    devices = [Device(0, "cpu")]
    for d in range(1, num_devices + 1):
        devices.append(Device(d, "gpu"))

    batch_size = 8
    hidden_dim = 4  # using this for input/output dim also

    input_vals = []
    for l in range(num_layers):
        for d in range(1, num_devices + 1):
            if l < num_layers // 2:
                shape = (hidden_dim, hidden_dim)
            else:
                shape = (hidden_dim, hidden_dim // num_devices)
            # w{l}_{d}:
            input_vals.append(Value("", Tensor(Float(), shape, devices[d])))
    for d in range(1, num_devices + 1):
        # x_{d}:
        shape = (batch_size // num_devices, hidden_dim)
        input_vals.append(Value("", Tensor(Float(), shape, devices[d])))

    # Test type inference:
    fn = infer_types(fn, input_vals)
    cpprint(fn)
    assert all(
        v.type.shape == (batch_size, hidden_dim // num_devices) for v in fn.outputs
    )

    # Test with sequential executor:
    np.random.seed(0)
    weights = [np.random.randn(hidden_dim, hidden_dim) for l in range(num_layers)]
    x = np.random.randn(batch_size, hidden_dim)

    # Split inputs for distributed function
    input_arrays = []
    for l in range(num_layers):
        if l < num_layers // 2:
            for d in range(1, num_devices + 1):
                input_arrays.append(weights[l])
        else:
            input_arrays += np.split(weights[l], num_devices, axis=1)
    input_arrays += np.split(x, num_devices)
    ex = SequentialExecutor("numpy")
    output_arrays = ex.compute(fn, input_arrays)

    # Expected results
    y = x
    for l in range(num_layers):
        y = np.matmul(y, weights[l])
    ys = np.split(y, num_devices, axis=1)
    assert all(np.allclose(y, o) for y, o in zip(ys, output_arrays))

    # Per-rank projection:
    per_rank_fns = project(fn, tuple(v.type for v in input_vals))
    for d, f_d in per_rank_fns.items():
        print()
        print(d)
        cpprint(f_d)

    # Make inputs for each per-rank function:
    per_rank_inputs = [[] for _ in range(num_devices)]
    for v, a in zip(fn.inputs, input_arrays):
        per_rank_inputs[v.type.device.device_id - 1].append(torch.tensor(a))

    # Run per-rank modules using PyTorch backend:
    per_rank_outputs = run_multiprocesses(per_rank_fns.values(), per_rank_inputs)

    # Check outputs:
    assert all(np.allclose(y, o) for y, o in zip(per_rank_outputs, output_arrays))


# TODO get DHP transform to work on mlp_inference and try running on backend
# def test_mlp_grid_search():
#     devices = [Device(d, "gpu") for d in range(3)]
#
#     f = mlp_inference(4, 6, 6, 6, 4, devices[0])
#     f = infer_types(f, f.inputs)
#
#     f_dist = mlp_dhp_transform(f, 2, 1, 1, devices, 1)


def test_dp_mlp():
    num_devices = 2
    num_layers = 4
    batch_size = 4
    hidden_dim = 6  # Also input/output dim for simplicity
    devices = [Device(d, "gpu") for d in range(num_devices + 1)]

    fn = mlp_inference_dp(
        batch_size, hidden_dim, hidden_dim, hidden_dim, num_layers, devices[1:]
    )
    fn = infer_types(fn, fn.inputs)
    cpprint(fn)

    def convert_inputs_dp(weights, x):
        xs = torch.split(x, num_devices)

        def new_inputs():
            for d in range(num_devices):
                yield from weights
                yield xs[d]

        return list(new_inputs())

    # Make random input/expected data:
    weights = [torch.randn(hidden_dim, hidden_dim) for _ in range(num_layers)]
    x = torch.randn(batch_size, hidden_dim)
    y = x
    for l in range(num_layers):
        y = torch.matmul(y, weights[l])
        y = torch.relu(y)

    # Project and run on backend:
    per_rank_fns = project(fn, tuple(v.type for v in fn.inputs))
    per_rank_inputs = [[] for _ in range(num_devices)]
    for v, a in zip(fn.inputs, convert_inputs_dp(weights, x)):
        per_rank_inputs[v.type.device.device_id - 1].append(a)
    per_rank_outputs, runtimes = run_multiprocesses(
        per_rank_fns.values(), per_rank_inputs
    )

    # Check outputs:
    assert torch.allclose(y, torch.cat(per_rank_outputs, 0))

    return runtimes


if __name__ == "__main__":
    # test_owt(2, 4)
    # test_dp_mlp()
    pass
