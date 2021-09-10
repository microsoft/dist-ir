from abc import ABC, abstractmethod
import csv
import itertools
from multiprocessing import Manager
from os import path
from typing import NamedTuple
import traceback

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from dist_ir.ir.topology import get_uniform_topology


FIELDNAMES = [
    "model_size",
    "world_size",
    "batch_size",
    "dp_degree",
    "hp_degree",
    "pp_degree",
    "num_microbatches",
    "latency",
    "throughput",
    "peak_memory",
]


class DHPConfig(NamedTuple):
    model_size: str
    dp_degree: int
    hp_degree: int
    pp_degree: int
    num_microbatches: int
    batch_size: int


class GridSearch(ABC):
    def __init__(
        self,
        model_params,
        backend,
        use_gpu,
        output_file,
        device_throughput,
        dram_bandwidth,
        kernel_launch_overhead,
        network_bandwidth,
        max_world_size,
        model_path=None,
    ):
        self.model_params = model_params
        self.backend = backend
        self.use_gpu = use_gpu
        self.output_file = output_file
        self.device_throughput = device_throughput
        self.dram_bandwidth = dram_bandwidth
        self.kernel_launch_overhead = kernel_launch_overhead
        self.network_bandwidth = network_bandwidth
        self.model_path = model_path
        self.topology = get_uniform_topology(
            max_world_size,
            self.device_throughput,
            self.dram_bandwidth,
            self.kernel_launch_overhead,
            self.network_bandwidth,
        )
        manager = Manager()
        self.lock = manager.Lock()

    def _write_row(self, config: DHPConfig, latency, peak_memory):
        throughput = config.batch_size / latency
        world_size = config.dp_degree * config.hp_degree * config.pp_degree
        with self.lock:
            with open(self.output_file, "a+", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow(
                    {
                        "model_size": config.model_size,
                        "world_size": world_size,
                        "batch_size": config.batch_size,
                        "dp_degree": config.dp_degree,
                        "hp_degree": config.hp_degree,
                        "pp_degree": config.pp_degree,
                        "num_microbatches": config.num_microbatches,
                        "latency": latency,
                        "throughput": throughput,
                        "peak_memory": peak_memory,
                    }
                )
                f.flush()

    # TODO is it cleaner to have these outside the class?
    @staticmethod
    def _config_from_df(df: pd.DataFrame, row_number):
        r = df.iloc[row_number]
        return DHPConfig(
            r.model_size,
            r.dp_degree,
            r.hp_degree,
            r.pp_degree,
            r.num_microbatches,
            r.batch_size,
        )

    @staticmethod
    def _read_configs(configs_file):
        df = pd.read_csv(configs_file)
        return [GridSearch._config_from_df(df, i) for i in range(len(df))]

    @staticmethod
    def _filter_configs_from_file(configs, file):
        """Filter `configs` to those configs that are not already in `file`."""
        existing_configs = set(GridSearch._read_configs(file))
        print(f"Found {len(existing_configs)} existing configurations, skipping them")
        return [c for c in configs if c not in existing_configs]

    @staticmethod
    def get_all_degrees(n):
        all_degrees = []
        d = 1
        h = 1
        p = 1
        while d <= n:
            h = 1
            p = 1
            if d * h * p == n:
                all_degrees.append((d, h, p))
                break
            while h <= n:
                p = 1
                if d * h * p == n:
                    all_degrees.append((d, h, p))
                    break
                while p <= n:
                    if d * h * p == n:
                        all_degrees.append((d, h, p))
                        break
                    p *= 2
                h *= 2
            d *= 2
        return all_degrees

    def gen_configurations(self, all_world_sizes, all_batch_sizes, all_model_sizes):
        for (
            world_size,
            batch_size,
            model_size,
        ) in itertools.product(all_world_sizes, all_batch_sizes, all_model_sizes):
            all_degrees = GridSearch.get_all_degrees(world_size)
            for (dp_degree, hp_degree, pp_degree) in all_degrees:
                dp_batch_size = batch_size // dp_degree
                if dp_batch_size == 0:
                    continue
                elif pp_degree == 1:
                    all_num_microbatches = [1]
                else:
                    all_num_microbatches = [
                        int(2 ** k)
                        for k in range(1, int(np.floor(np.log2(dp_batch_size) / 2)))
                    ]
                for num_microbatches in all_num_microbatches:
                    config = DHPConfig(
                        model_size,
                        dp_degree,
                        hp_degree,
                        pp_degree,
                        num_microbatches,
                        batch_size,
                    )
                    try:
                        self.verify_config(config)
                    except Exception as e:
                        print(f"Skipping configuration {config}:\n{e}")
                        continue

                    yield config

    def get_model_params(self, model_size):
        return self.model_params[model_size]

    @abstractmethod
    def get_model_and_input_data(self, batch_size, model_size):
        pass

    @abstractmethod
    def verify_config(self, config: DHPConfig):
        pass

    @abstractmethod
    def transform(
        self,
        fn,
        input_data,
        topology,
        config: DHPConfig,
    ):
        pass

    @abstractmethod
    def simulate(transformed_fn, input_data, topology):
        pass

    @abstractmethod
    def pytorch(transformed_fn, input_data, world_size):
        pass

    def run(self, config: DHPConfig):
        fn, input_data = self.get_model_and_input_data(
            config.batch_size, config.model_size
        )
        try:
            _, transformed_fn, input_data = self.transform(
                fn, input_data, self.topology, config
            )
            if self.backend == "simulate":
                simulation = self.simulate(transformed_fn, input_data, self.topology)
                latency = max([simulation.timestamps[d] for d in simulation.timestamps])
                peak_memory = max(
                    [simulation.peak_memory[d] for d in simulation.peak_memory]
                ) / (2.0 ** 20)
            elif self.backend == "pytorch":
                world_size = config.dp_degree * config.hp_degree * config.pp_degree
                _, runtimes = self.pytorch(transformed_fn, input_data, world_size)
                latency = np.median(runtimes[-1])
                # TODO: Measure peak memory?
                peak_memory = 0
        except Exception as e:
            print(f"Failed to run the configuration {config}:")
            traceback.print_exc()

            latency = -1
            peak_memory = -1
        except RuntimeError as e:
            print(e)
            latency = -1
            peak_memory = -1
        self._write_row(config, latency, peak_memory)

    def grid_search(self, configs):
        if self.backend == "pytorch":
            for config in configs:
                print(config)
                self.run(config)
        elif self.backend == "simulate":
            process_map(self.run, configs)
        else:
            raise ValueError(f"Invalid backend {self.backend}")


# TODO merge with grid_search? move everything there or here?
def run_grid_search(args, grid_search_cls):
    grid_search = grid_search_cls(
        args.backend,
        args.use_gpu,
        args.output_file,
        args.device_throughput,
        args.dram_bandwidth,
        args.kernel_launch_overhead,
        args.network_bandwidth,
        max(args.all_world_sizes),
        model_path=args.model_path if hasattr(args, "model_path") else None,
    )

    # If we are not given which config(s) to run, generate them
    if args.configs_file is None:
        configs = list(
            grid_search.gen_configurations(
                args.all_world_sizes, args.all_batch_sizes, args.all_model_sizes
            )
        )
        print(f"Generated {len(configs)} configurations")
    else:
        if args.config_number is not None:
            df = pd.read_csv(args.configs_file)
            # lookup and run only given config
            configs = [GridSearch._config_from_df(df, args.config_number - 1)]
        else:
            # use all configs
            configs = GridSearch._read_configs(args.configs_file)
        print(f"Found {len(configs)} configurations")

    # If output file exists, skip existing configs and append results to output file
    if path.exists(args.output_file) and not args.overwrite_output_file:
        if args.append_output_file:
            print(f'File "{args.output_file}" already exists. Appending to it')
        else:
            message = f'File "{args.output_file}" already exists. Append to it? [y/n] '
            if input(message).lower().strip()[0] != "y":
                return

        configs = GridSearch._filter_configs_from_file(configs, args.output_file)
    else:
        with open(args.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

    grid_search.grid_search(configs)
