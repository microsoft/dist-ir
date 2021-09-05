from argparse import ArgumentParser


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
        parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    def add_simulation_topology_config_arguments(self):
        self.add_argument(
            "--network_bandwidth",
            type=float,
            default=64,
            help="Network bandwidth in Gbps",
        )
        self.add_argument(
            "--device_throughput", type=float, default=1.4e13, help="Device throughput"
        )
        self.add_argument(
            "--dram_bandwidth", type=float, default=9e11, help="DRAM Bandwidth"
        )
        self.add_argument(
            "--kernel_launch_overhead",
            type=float,
            default=1e-5,
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
            "--use-gpu",
            action="store_true",
            default=False,
            help="Use GPU with PyTorch backend",
        )

    def add_grid_search_output_config_arguments(self):
        self.add_argument(
            "--output_file",
            type=str,
            required=True,
            help="Output file",
        )

    def add_calibration_arguments(self):
        # TODO: Add for simulator accuracy
        pass
