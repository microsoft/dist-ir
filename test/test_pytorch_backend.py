import numpy as np
import pytest
import torch

from dist_ir.backend.torch import run_pytorch
from dist_ir.executor import sequentially_execute
from dist_ir.executor.concrete_value import ConcreteValue
from dist_ir.executor.cost_model import CostModel
from dist_ir.executor.simulator import Simulator
from dist_ir.executor.type_inference import infer_types
from dist_ir.ir import Device, FunctionMaker, cpprint, Value
from dist_ir.ir.type import Float16, Float32, Tensor
from dist_ir.ir.topology import Topology, get_uniform_topology

# TODO make examples submodule of dist_ir?
from examples.mlp import mlp, mlp_inference_dp

torch.manual_seed(42)


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
        attributes={"axis": 0},
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
                attributes={"axis": 1},
            )

    fn.set_outputs(hs)
    return fn.finalize()


@pytest.mark.parametrize(
    "num_devices, num_layers, use_gpu, dtype",
    [
        (2, 4, False, "fp32"),
        pytest.param(
            2,
            4,
            True,
            "fp32",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="Not enough available GPUs"
            ),
        ),
        pytest.param(
            2,
            4,
            True,
            "fp16",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="Not enough available GPUs"
            ),
        ),
    ],
)
def test_owt(num_devices, num_layers, use_gpu, dtype):
    dist_ir_dtype = Float32 if dtype == "fp32" else Float16
    numpy_dtype = np.float32 if dtype == "fp32" else np.float16
    torch_dtype = torch.float32 if dtype == "fp32" else torch.float16

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
            input_vals.append(Value("", Tensor(dist_ir_dtype(), shape, devices[d])))
    for d in range(1, num_devices + 1):
        # x_{d}:
        shape = (batch_size // num_devices, hidden_dim)
        input_vals.append(Value("", Tensor(dist_ir_dtype(), shape, devices[d])))

    # Test type inference:
    fn = infer_types(fn, input_vals)
    cpprint(fn)
    assert all(
        v.type.shape == (batch_size, hidden_dim // num_devices) for v in fn.outputs
    )

    # Test with sequential executor:
    np.random.seed(0)
    weights = [
        np.random.normal(0, 0.02, size=(hidden_dim, hidden_dim)).astype(numpy_dtype)
        for l in range(num_layers)
    ]
    x = np.random.normal(0, 0.02, size=(batch_size, hidden_dim)).astype(numpy_dtype)

    # Split inputs for distributed function
    input_arrays = []
    for l in range(num_layers):
        if l < num_layers // 2:
            for d in range(1, num_devices + 1):
                input_arrays.append(weights[l])
        else:
            input_arrays += np.split(weights[l], num_devices, axis=1)
    input_arrays += np.split(x, num_devices)
    inputs = [ConcreteValue(v, None) for v in input_arrays]
    outputs = sequentially_execute(fn, inputs)
    output_arrays = [v.val for v in outputs]

    # Expected results
    y = x
    for l in range(num_layers):
        y = np.matmul(y, weights[l])
    ys = np.split(y, num_devices, axis=1)
    assert all(np.allclose(y, o) for y, o in zip(ys, output_arrays))

    # Run per-rank modules using PyTorch backend:
    results = run_pytorch(
        fn, [torch.tensor(a).to(torch_dtype) for a in input_arrays], use_gpu=use_gpu
    )

    # Check outputs:
    assert all(
        np.allclose(y[0], o) for y, o in zip(results.per_rank_outputs, output_arrays)
    )


def test_dp_mp_matmuls():
    fn = FunctionMaker("dp_mp_matmuls")
    B = 64
    d0 = Device(0, "gpu")
    d1 = Device(1, "gpu")
    x_0 = fn.add_input_value("x_0", Tensor(Float32(), (B // 2, B), d0))
    x_1 = fn.add_input_value("x_1", Tensor(Float32(), (B // 2, B), d1))
    wA_0 = fn.add_input_value("wA_0", Tensor(Float32(), (B, B), d0))
    wA_1 = fn.add_input_value("wA_1", Tensor(Float32(), (B, B), d1))
    wB_0 = fn.add_input_value("wB_0", Tensor(Float32(), (B, B), d0))
    wC_1 = fn.add_input_value("wC_1", Tensor(Float32(), (B, B), d1))
    a0_0 = fn.add_op("MatMul", inputs=[x_0, wA_0], output_names=["a0"])
    a1_1 = fn.add_op("MatMul", inputs=[x_1, wA_1], output_names=["a1"])
    a_0 = fn.add_op(
        "MPIGather",
        inputs=[a0_0, a1_1],
        output_names=["a_0"],
        attributes={"device": d0, "axis": 0},
    )
    b_0 = fn.add_op("MatMul", inputs=[a_0, wB_0], output_names=["b_0"])
    b_1 = fn.add_op(
        "Send", inputs=[b_0], output_names=["b_1"], attributes={"device": d1}
    )
    c_1 = fn.add_op("MatMul", inputs=[b_1, wC_1], output_names=["c_1"])
    fn = fn.finalize()
    fn = infer_types(fn, fn.inputs)
    cpprint(fn)

    from dist_ir.executor.rank_projector import project

    per_rank_fns, groups = project(fn, tuple(v.type for v in fn.inputs))
    for per_rank_fn in per_rank_fns.values():
        cpprint(per_rank_fn)


@pytest.mark.parametrize(
    "use_gpu",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 1, reason="Not enough available GPUs"
            ),
        ),
    ],
)
def test_single_device(use_gpu):
    d1 = Device(1, "gpu")
    fn = FunctionMaker()
    x = fn.add_input_value("x", Tensor(Float32(), (4, 4), d1))
    y = fn.add_op("MatMul", inputs=(x, x))
    fn.set_outputs((y,))
    fn = fn.finalize()
    cpprint(fn)

    x = torch.randn(4, 4)
    inputs = (x,)
    results = run_pytorch(fn, inputs, use_gpu=use_gpu)
    print(results.per_rank_outputs)
    assert torch.allclose(torch.matmul(x, x), results.per_rank_outputs[0][0])


@pytest.mark.parametrize(
    "use_gpu",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="Not enough available GPUs"
            ),
        ),
    ],
)
def test_send_recv(use_gpu):
    d1 = Device(1, "gpu")
    d2 = Device(2, "gpu")
    fn = FunctionMaker()
    x = fn.add_input_value("x", Tensor(Float32(), (4, 4), d1))
    y = fn.add_op("Send", inputs=(x,), attributes={"device": d2})
    fn.set_outputs((x, y))
    fn = fn.finalize()
    cpprint(fn)

    x = torch.randn(4, 4)
    inputs = (x,)
    results = run_pytorch(fn, inputs, use_gpu=use_gpu)
    assert torch.allclose(x, results.per_rank_outputs[1][0])


@pytest.mark.parametrize(
    "use_gpu, dtype",
    [
        (False, "fp32"),
        (False, "fp16"),
        pytest.param(
            True,
            "fp16",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="Not enough available GPUs"
            ),
        ),
        pytest.param(
            True,
            "fp32",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="Not enough available GPUs"
            ),
        ),
    ],
)
def test_dp_mlp(use_gpu, dtype):
    num_devices = 2
    num_layers = 4
    batch_size = 4
    hidden_dim = 6  # Also input/output dim for simplicity
    devices = [Device(d, "gpu") for d in range(num_devices + 1)]

    fn = mlp_inference_dp(
        batch_size,
        hidden_dim,
        hidden_dim,
        hidden_dim,
        num_layers,
        devices[1:],
        Float32 if dtype == "fp32" else Float16,
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
    results = run_pytorch(
        fn,
        convert_inputs_dp(weights, x),
        use_gpu=use_gpu,
    )

    # Check outputs:
    assert torch.allclose(y, torch.cat([o[0] for o in results.per_rank_outputs], 0))

    return results.latency


def test_separate_projection_types():
    d1 = Device(1, "gpu")
    d2 = Device(2, "gpu")
    fn = FunctionMaker()
    x = fn.add_input_value("x", None)
    y = fn.add_op("Send", inputs=(x,), attributes={"device": d2})
    fn.set_outputs((x, y))
    fn = fn.finalize()
    cpprint(fn)

    x = torch.randn(4, 4)
    inputs = [x]
    input_types = [Tensor(Float32(), (4, 4), d1)]
    results = run_pytorch(fn, inputs, input_types=input_types)
    assert torch.allclose(x, results.per_rank_outputs[1][0])

    x = torch.randn(8, 8)
    inputs = [x]
    input_types = [Tensor(Float32(), (8, 8), d1)]
    results = run_pytorch(fn, inputs, input_types=input_types)
    assert torch.allclose(x, results.per_rank_outputs[1][0])


if __name__ == "__main__":
    # test_owt(2, 4, use_gpu=False)
    # test_dp_mlp(use_gpu=False)
    # test_send_recv(use_gpu=False)
    # test_single_device(use_gpu=False)
    test_dp_mp_matmuls()
    test_mlp_grid_search(use_gpu=False)
