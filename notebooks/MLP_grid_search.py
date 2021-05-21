# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

from collections import defaultdict
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
import torch

from dist_ir.backend.torch import run_pytorch
from dist_ir.executor import SequentialExecutor
from dist_ir.executor.cost_model import CostModel
from dist_ir.executor.simulator import Simulator
from dist_ir.executor.type_inference import infer_types
from dist_ir.ir import Device, FunctionMaker, cpprint, Value
from dist_ir.ir.type import Float, Tensor
from dist_ir.ir.topology import Topology

# TODO notebook server needs to be run from dist-ir/
from examples.grid_search import add_devices_to_topology, gen_configurations, mlp_dist
from examples.mlp import mlp, mlp_inference_dp


# %%

results = pd.read_pickle("../mlp_grid_search.pkl")
# results = results.astype({"simulated_time": float, "actual_time": float})
results

# %%

# Temp code to add simulated times from pkl file
sim_times = pd.read_pickle("../mlp_simulated.pkl")
results["simulated_time"] = sim_times["simulated_time"]
results

# # Temp code to add actual times from stdout dump
# # mlp_grid_search.pkl already has actual times from stdout dump
# with open("../grid_search.out", "r") as fin:
#     raw_times = fin.readlines()
# i = 0
# for line in raw_times:
#     line = line.split(" ")
#     if len(line) == 3:
#         results.at[i, "actual_time"] = float(line[2].strip())
#         i += 1
# results = results[:i]
# results

# %%


def plot_throughputs(results):
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.autolimit_mode"] = "round_numbers"
    lines = []
    labels = ["Ideal", "Best fit"]

    simulated_throughputs = np.array(
        results["batch_size"] / results["simulated_time"] / 1000
    )
    real_throughputs = np.array(results["batch_size"] / results["actual_time"] / 1000)
    r, p = pearsonr(simulated_throughputs, real_throughputs)
    print(f"Pearson's correlation: {r} (p={p})")
    r, p = spearmanr(simulated_throughputs, real_throughputs)
    print(f"Spearman's correlation: {r} (p={p})")

    x_new = np.linspace(
        min(simulated_throughputs.min(), real_throughputs.min()),
        max(simulated_throughputs.max(), real_throughputs.max()),
        500,
    )
    lines.append(
        plt.plot(x_new, x_new, color="black", linestyle="--", label="Ideal")[0]
    )
    m, b = np.polyfit(simulated_throughputs, real_throughputs, 1)
    f = interp1d(simulated_throughputs, m * simulated_throughputs + b, kind="linear")
    x_new = np.linspace(simulated_throughputs.min(), simulated_throughputs.max(), 500)
    y_smooth = f(x_new)
    lines.append(
        plt.plot(x_new, y_smooth, color="orange", linestyle="-.", label="Best fit")[0]
    )
    colors = ["b", "orange", "g", "purple"]
    markers = ["x", "o", "^"]
    plt.scatter(simulated_throughputs, real_throughputs, marker="x")
    plt.grid()
    # plt.xticks([0, 200, 400, 600, 800, 1000])
    # plt.yticks([0, 200, 400, 600, 800, 1000])
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
    # plt.savefig(
    #     "data_parallel_simulation_performance.pdf", dpi=600, bbox_inches="tight"
    # )


# %%

plot_throughputs(results)

# %%

results["err"] = results["actual_time"] - results["simulated_time"]
results.sort_values(by="err", ascending=False)

# TODO
# - dump chrome trace from pytorch
# - See if matmul timings, sequential timings match up
