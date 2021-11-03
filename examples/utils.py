import json
import logging


def configure_logging(args):
    # TODO: Make logging level a command line arg
    if args.loglevel == "info":
        level = logging.INFO
    elif args.loglevel == "debug":
        level = logging.DEBUG
    elif args.loglevel == "warning":
        level = logging.WARNING
    elif args.loglevel == "error":
        level = logging.ERROR
    elif args.loglevel == "critical":
        level = logging.CRITICAL
    logging.basicConfig(filename=args.logname, level=level)
    logging.getLogger().addHandler(logging.StreamHandler())


def load_simulation_parameters_to_args(args):
    if args.simulation_parameters_file is not None:
        with open(args.simulation_parameters_file, "r") as f:
            simulation_parameters = json.load(f)
        args.dram_bandwidth = (
            float("inf")
            if simulation_parameters["device_parameters"][0] == 0
            else 1.0 / simulation_parameters["device_parameters"][0]
        )
        args.device_throughput = 1.0 / simulation_parameters["device_parameters"][1]
        args.kernel_launch_overhead = simulation_parameters["device_parameters"][2]
        args.network_bandwidth = simulation_parameters["network_bandwidth"]
        args.allreduce_parameters = {
            int(k): v for k, v in simulation_parameters["allreduce_parameters"].items()
        }
