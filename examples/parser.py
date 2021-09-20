from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch

from dist_ir.utils import constants


class Parser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs["formatter_class"] = ArgumentDefaultsHelpFormatter
        super().__init__(*args, **kwargs)

    def parse_args(self):
        args = super().parse_args()

        # Check arguments are valid:
        if hasattr(args, "mode"):
            assert args.mode is not None
            if args.mode == "file":
                assert args.configs_file is not None
            elif args.mode == "config":
                assert args.config is not None
                assert args.model_size is not None

        return args

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

    def add_simulation_config_arguments(self):
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
        self.add_argument(
            "--allreduce_parameters", default=None, help="Allreduce parameters"
        )
        self.add_argument(
            "--simulation_parameters_file",
            type=str,
            default=None,
            help="Simulation parameters file",
        )

    def add_execution_mode_config_arguments(self):
        self.add_argument("--backend", choices=["simulate", "pytorch"], required=True)
        self.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")

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
        # 3 modes: generate & search grid, run from file, run single config
        self.add_argument(
            "--mode",
            type=str,
            choices=["grid", "file", "config"],
            default=None,
            help="Run mode: run a grid search, run a single/all configs from a"
            "file, run a specified config from command line",
        )
        # grid search arguments:
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
        # config file arguments:
        self.add_argument(
            "--configs_file",
            type=str,
            default=None,
            help="File containing configurations to run",
        )
        self.add_argument(
            "--config_number",
            type=int,
            default=None,
            help="The configuration from configs_file to run (line number, 0 = header)",
        )
        # single config arguments:
        self.add_argument(
            "--model_size",
            type=str,
            default=None,
            help="The model size to run when mode == config, e.g. mlp-xs or gpt3",
        )
        self.add_argument(
            "--config",
            nargs="+",
            type=int,
            default=None,
            help="The configration to run (D H P K BS, e.g.: --config 1 2 2 8 128)",
        )
        # output file arguments:
        self.add_argument(
            "--output_file",
            type=str,
            required=True,
            help="Output file",
        )
        output_file_group = self.add_mutually_exclusive_group()
        output_file_group.add_argument(
            "--append_output_file",
            action="store_true",
            default=False,
            help="Append to output file (and skip configurations already present)",
        )
        output_file_group.add_argument(
            "--overwrite_output_file",
            action="store_true",
            default=False,
            help="Overwrite output file",
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
