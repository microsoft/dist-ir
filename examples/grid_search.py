from abc import ABC, abstractmethod
import csv
import copy
import itertools
from multiprocessing import Manager
import numpy as np
from tqdm.contrib.concurrent import process_map
import traceback

from dist_ir.ir import get_uniform_topology

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
    ):
        self.model_params = model_params
        self.backend = backend
        self.use_gpu = use_gpu
        self.output_file = output_file
        self.device_throughput = device_throughput
        self.dram_bandwidth = dram_bandwidth
        self.kernel_launch_overhead = kernel_launch_overhead
        self.network_bandwidth = network_bandwidth

    def _write_row(self, config, latency, peak_memory):
        (
            topology,
            world_size,
            batch_size,
            model_size,
            dp_degree,
            hp_degree,
            pp_degree,
            num_microbatches,
            lock,
        ) = config
        throughput = batch_size / latency
        with lock:
            with open(self.output_file, "a+", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow(
                    {
                        "model_size": model_size,
                        "world_size": world_size,
                        "batch_size": batch_size,
                        "dp_degree": dp_degree,
                        "hp_degree": hp_degree,
                        "pp_degree": pp_degree,
                        "num_microbatches": num_microbatches,
                        "latency": latency,
                        "throughput": throughput,
                        "peak_memory": peak_memory,
                    }
                )
                f.flush()

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

    def gen_configurations(
        self, topology, all_world_sizes, all_batch_sizes, all_model_sizes
    ):
        manager = Manager()
        lock = manager.Lock()
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
                    try:
                        self.verify_config(
                            batch_size,
                            dp_degree,
                            hp_degree,
                            pp_degree,
                            num_microbatches,
                            model_size,
                        )
                    except Exception as e:
                        print(
                            f"Skipping configuration batch_size={batch_size}, "
                            f"model_size={model_size}, dp_degree={dp_degree}, "
                            f"hp_degree={hp_degree}, pp_degree={pp_degree}, "
                            f"num_microbatches={num_microbatches}: {e}"
                        )
                        continue

                    yield (
                        topology,
                        world_size,
                        batch_size,
                        model_size,
                        dp_degree,
                        hp_degree,
                        pp_degree,
                        num_microbatches,
                        lock,
                    )

    def get_model_params(self, model_size):
        return self.model_params[model_size]

    @abstractmethod
    def prepare_models_and_input_data(self, topology, all_batch_sizes, all_model_sizes):
        pass

    @abstractmethod
    def get_model_and_input_data(self, model_size, batch_size):
        pass

    @abstractmethod
    def verify_config(
        self, batch_size, dp_degree, hp_degree, pp_degree, num_microbatches, model_size
    ):
        pass

    @abstractmethod
    def transform(
        self,
        fn,
        input_data,
        topology,
        dp_degree,
        hp_degree,
        pp_degree,
        num_microbatches,
        model_size,
    ):
        pass

    @abstractmethod
    def simulate(transformed_fn, input_data, topology):
        pass

    @abstractmethod
    def pytorch(transformed_fn, input_data, world_size):
        pass

    def run(self, config):
        (
            topology,
            world_size,
            batch_size,
            model_size,
            dp_degree,
            hp_degree,
            pp_degree,
            num_microbatches,
            lock,
        ) = config
        fn, input_data = self.get_model_and_input_data(batch_size, model_size)
        try:
            init_fn, transformed_fn, input_data = self.transform(
                fn,
                input_data,
                topology,
                dp_degree,
                hp_degree,
                pp_degree,
                num_microbatches,
                model_size,
            )
            if self.backend == "simulate":
                simulation = self.simulate(transformed_fn, input_data, topology)
                latency = max([simulation.timestamps[d] for d in simulation.timestamps])
                peak_memory = max(
                    [simulation.peak_memory[d] for d in simulation.peak_memory]
                ) / (2.0 ** 20)
            elif self.backend == "pytorch":
                world_size = len(topology.devices) - 1
                per_rank_outputs, runtimes = self.pytorch(
                    transformed_fn, input_data, world_size
                )
                latency = np.median(runtimes[-1])
                # TODO: Measure peak memory?
                peak_memory = 0
        except Exception as e:
            print(
                f"Failed to run the configuration model_size={model_size}, "
                f"batch_size={batch_size}, dp_degree={dp_degree}, "
                f"hp_degree={hp_degree}, pp_degree={pp_degree}, "
                f"num_microbatches={num_microbatches}:"
            )
            traceback.print_exc()

            latency = -1
            peak_memory = -1
        self._write_row(config, latency, peak_memory)

    def grid_search(self, all_world_sizes, all_batch_sizes, all_model_sizes):
        topology = get_uniform_topology(
            max(all_world_sizes),
            self.device_throughput,
            self.dram_bandwidth,
            self.kernel_launch_overhead,
            self.network_bandwidth,
        )

        self.prepare_models_and_input_data(topology, all_batch_sizes, all_model_sizes)
        configs = list(
            self.gen_configurations(
                topology, all_world_sizes, all_batch_sizes, all_model_sizes
            )
        )
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        if self.backend == "pytorch":
            process_map(self.run, configs, max_workers=1)
        elif self.backend == "simulate":
            process_map(self.run, configs)
        else:
            raise ValueError(f"Invalid backend {backend}")
