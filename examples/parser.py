from argparse import ArgumentParser
import torch

from dist_ir.utils import constants


class Parser(ArgumentParser):
    def add_parallelism_config_arguments(self):
        self.add_argument(
            "-d", "--dp_degree", type=int, default=1, help="Data parallel degree"
        )
        self.add_argument(
            "-t", "--hp_degree", type=int, default=1, help="Horizontal parallel degree"
        )
        self.add_argument(
            "-p", "--pp_degree", type=int, default=1, help="Pipeline parallel degree"
        )
        self.add_argument(
            "-k", "--num_microbatches", type=int, default=1, help="# of microbatches"
        )
        self.add_argument("--batch_size", type=int, default=64, help="Batch size")

    def add_simulation_topology_config_arguments(self):
        self.add_argument(
            "--network_bandwidth",
            type=float,
            default=constants.DEFAULT_NETWORK_BANDWIDTH,
            help="Network bandwidth in Gbps",
        )
        self.add_argument(
            "--device_throughput",
            type=float,
            default=constants.DEFAULT_DEVICE_THROUGHPUT,
            help="Device throughput",
        )
        self.add_argument(
            "--dram_bandwidth",
            type=float,
            default=constants.DEFAULT_DRAM_BANDWIDTH,
            help="DRAM Bandwidth",
        )
        self.add_argument(
            "--kernel_launch_overhead",
            type=float,
            default=constants.DEFAULT_KERNEL_LAUNCH_OVERHEAD,
            help="Kernel launch overhead",
        )

    def add_execution_mode_config_arguments(self):
        self.add_argument("--backend", choices=["simulate", "pytorch"])

    def add_simulation_output_config_arguments(self):
        self.add_argument("--trace_file", type=str, default=None, help="Trace file")

    def add_backend_config_arguments(self):
        self.add_argument(
            "--debug_stacktrace",
            default=False,
            action="store_true",
            help="Debug stacktrace",
        )
        self.add_argument(
            "--use_gpu",
            action="store_true",
            default=torch.cuda.is_available(),
            help="Use GPU with PyTorch backend",
        )

    def add_grid_search_config_arguments(self, defaults):
        self.add_argument(
            "--all_world_sizes",
            nargs="+",
            type=int,
            default=defaults["all_world_sizes"],
        )
        self.add_argument(
            "--all_batch_sizes",
            nargs="+",
            type=int,
            default=defaults["all_batch_sizes"],
        )
        self.add_argument(
            "--all_model_sizes",
            nargs="+",
            type=str,
            default=defaults["all_model_sizes"],
        )
        self.add_argument(
            "--simulation_results_file",
            type=str,
            default=None,
            help="Simulation results file",
        )
        self.add_argument(
            "--output_file",
            type=str,
            required=True,
            help="Output file",
        )

    def add_global_output_config_arguments(self):
        self.add_argument(
            "--verbose", action="store_true", default=False, help="Verbose"
        )

    def add_gpt2_model_path_config_arguments(self):
        self.add_argument(
            "--model_path",
            type=str,
            required=True,
            help=(
                "Path to GPT-2 ONNX model "
                "(downloaded from https://github.com/onnx/models/blob/master/"
                "text/machine_comprehension/gpt-2/model/gpt2-10.onnx?raw=True)"
            ),
        )

    def add_calibration_arguments(self):
        # TODO: Add for simulator accuracy
        pass
