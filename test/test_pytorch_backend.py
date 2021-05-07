from collections import defaultdict
import csv
import numpy as np
import pytest
import torch

from dist_ir.backend.torch import run_pytorch
from dist_ir.executor import SequentialExecutor
from dist_ir.executor.cost_model import CostModel
from dist_ir.executor.simulator import Simulator
from dist_ir.executor.type_inference import infer_types
from dist_ir.ir import Device, FunctionMaker, cpprint, Value
from dist_ir.ir.type import Float, Tensor
from dist_ir.ir.topology import Topology

# TODO make examples submodule of dist_ir?
from examples.grid_search import add_devices_to_topology, gen_configurations, mlp_dist
from examples.mlp import mlp, mlp_inference_dp


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
        attributes={"dim": 0},
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
                attributes={"dim": 1},
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

    # Run per-rank modules using PyTorch backend:
    per_rank_outputs, _ = run_pytorch(
        num_devices, fn, [torch.tensor(a) for a in input_arrays]
    )

    # Check outputs:
    assert all(np.allclose(y[0], o) for y, o in zip(per_rank_outputs, output_arrays))


def test_mlp_grid_search():
    batch_size = 2 ** 10
    hidden_dim = batch_size
    num_layers = 8
    world_size = 2

    topology = Topology()
    d0 = topology.add_device("gpu")
    add_devices_to_topology(topology, world_size)
    simulator = Simulator(CostModel(topology))
    seq_executor = SequentialExecutor("numpy")

    seq_mlp = mlp(batch_size, hidden_dim, hidden_dim, hidden_dim, num_layers, d0)
    seq_mlp = infer_types(seq_mlp, seq_mlp.inputs)
    configs = list(
        gen_configurations([hidden_dim], [world_size], [num_layers], [batch_size])
    )
    dist_mlp_fns = [
        mlp_dist(seq_mlp, d, h, p, m, topology) for (_, _, _, d, h, p, m) in configs
    ]
    print(len(dist_mlp_fns))

    # Create random input data
    input_data = tuple(
        np.random.randn(*v.type.shape).astype(np.float32) for v in seq_mlp.inputs
    )

    results = []
    for init_fn, fn in dist_mlp_fns:
        # Simulate
        simulation = simulator.interpret(fn, (v.type for v in fn.inputs))
        simulated_time = max([simulation.timestamps[d] for d in simulation.timestamps])

        # Reference-execute init_fn to get inputs for fn
        dist_input_data = seq_executor.compute(init_fn, input_data)
        dist_input_data = tuple(torch.tensor(t) for t in dist_input_data)
        assert all(
            t.shape == v.type.shape for (t, v) in zip(dist_input_data, fn.inputs)
        )

        # Measure actual execution time
        # TODO check outputs match?
        _, runtimes = run_pytorch(world_size, fn, dist_input_data)
        actual_time = max(np.median(times) for times in runtimes)

        print(fn.name, simulated_time, actual_time)
        results.append(
            (
                world_size,
                num_layers,
                batch_size,
                hidden_dim,
                simulated_time,
                actual_time,
            )
        )

    print(len(dist_mlp_fns))

    fieldnames = [
        "world_size",
        "num_layers",
        "batch_size",
        "hidden_dim",
        "simulated_time",
        "actual_time",
    ]

    with open("mlp_grid_search.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (
            world_size,
            num_layers,
            batch_size,
            hidden_dim,
            simulated_time,
            actual_time,
        ) in results:
            writer.writerow(
                {
                    "world_size": world_size,
                    "num_layers": num_layers,
                    "batch_size": batch_size,
                    "hidden_dim": hidden_dim,
                    "simulated_time": simulated_time,
                    "actual_time": actual_time,
                }
            )


def plot_mlp_grid_search_results():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    from scipy.stats import pearsonr, spearmanr

    results = []
    with open("mlp_grid_search.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                (
                    int(row["world_size"]),
                    int(row["num_layers"]),
                    int(row["batch_size"]),
                    int(row["hidden_dim"]),
                    float(row["simulated_time"]),
                    float(row["actual_time"]),
                )
            )
    real_throughputs = defaultdict(list)
    simulated_throughputs = defaultdict(list)
    for world_size, _, batch_size, _, simulated_time, actual_time in results:
        real_throughputs[world_size].append(batch_size / actual_time / 1000)
        simulated_throughputs[world_size].append(batch_size / simulated_time / 1000)
    plt.rcParams["font.size"] = 12
    all_simulated_throughputs = []
    all_real_throughputs = []
    lines = []
    labels = ["Ideal", "Best fit"]
    for world_size in simulated_throughputs:
        all_real_throughputs += real_throughputs[world_size]
    for world_size in simulated_throughputs:
        all_simulated_throughputs += simulated_throughputs[world_size]
    all_simulated_throughputs = np.array(all_simulated_throughputs)
    all_real_throughputs = np.array(all_real_throughputs)
    r, p = pearsonr(all_simulated_throughputs, all_real_throughputs)
    print(f"Pearson's correlation: {r} (p={p})")
    r, p = spearmanr(all_simulated_throughputs, all_real_throughputs)
    print(f"Spearman's correlation: {r} (p={p})")
    x_new = np.linspace(
        min(all_simulated_throughputs.min(), all_real_throughputs.min()),
        max(all_simulated_throughputs.max(), all_real_throughputs.max()),
        500,
    )
    lines.append(
        plt.plot(x_new, x_new, color="black", linestyle="--", label="Ideal")[0]
    )
    m, b = np.polyfit(all_simulated_throughputs, all_real_throughputs, 1)
    f = interp1d(
        all_simulated_throughputs, m * all_simulated_throughputs + b, kind="linear"
    )
    x_new = np.linspace(
        all_simulated_throughputs.min(), all_simulated_throughputs.max(), 500
    )
    y_smooth = f(x_new)
    lines.append(
        plt.plot(x_new, y_smooth, color="orange", linestyle="-.", label="Best fit")[0]
    )
    colors = ["b", "orange", "g", "purple"]
    markers = ["x", "o", "^"]
    plt.scatter(all_simulated_throughputs, all_real_throughputs, marker="x")
    plt.grid()
    plt.xticks([0, 200, 400, 600, 800, 1000])
    plt.yticks([0, 200, 400, 600, 800, 1000])
    plt.xlabel("Simulated throughput\n(1000 samples / second)")
    plt.ylabel("Real throughput\n(1000 samples / second)")
    plt.gca().set_aspect("equal", adjustable="box")
    leg = plt.figlegend(lines, labels, loc="upper center", ncol=2)
    leg.get_frame().set_linewidth(0.0)
    bb = leg.get_bbox_to_anchor().transformed(plt.gca().transAxes.inverted())
    yOffset = 0
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg.set_bbox_to_anchor(bb, transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(
        "data_parallel_simulation_performance.pdf", dpi=600, bbox_inches="tight"
    )


def test_empty_device():
    d1 = Device(1, "gpu")
    d2 = Device(2, "gpu")
    fn = FunctionMaker()
    x = fn.add_input_value("x", Tensor(Float(), (4, 4), d1))
    y = fn.add_op("MatMul", inputs=(x, x))
    fn.set_outputs((y,))
    fn = fn.finalize()
    cpprint(fn)

    x = torch.randn(4, 4)
    inputs = (x,)
    outputs, _ = run_pytorch(2, fn, inputs)
    print(outputs)
    assert torch.allclose(torch.matmul(x, x), outputs[0][0])


def test_send_recv():
    d1 = Device(1, "gpu")
    d2 = Device(2, "gpu")
    fn = FunctionMaker()
    x = fn.add_input_value("x", Tensor(Float(), (4, 4), d1))
    y = fn.add_op("Send", inputs=(x,), attributes={"device": d2})
    fn.set_outputs((x, y))
    fn = fn.finalize()
    cpprint(fn)

    x = torch.randn(4, 4)
    inputs = (x,)
    outputs, _ = run_pytorch(2, fn, inputs)
    assert torch.allclose(x, outputs[1][0])


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
    per_rank_outputs, runtimes = run_pytorch(
        num_devices, fn, convert_inputs_dp(weights, x)
    )

    # Check outputs:
    assert torch.allclose(y, torch.cat([o[0] for o in per_rank_outputs], 0))

    return runtimes


if __name__ == "__main__":
    # test_owt(2, 4)
    # test_dp_mlp()
    # test_send_recv()
    # test_empty_device()

    # import logging
    # logging.basicConfig(level=logging.INFO)
    test_mlp_grid_search()
    plot_mlp_grid_search_results()
