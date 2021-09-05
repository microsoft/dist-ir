from itertools import product
import numpy as np
import argparse

from dist_ir.executor import infer_types, SequentialExecutor
from dist_ir.transforms import mlp_dhp_transform
from . import mlp
from .grid_search import GridSearch
from .parser import Parser

MODEL_PARAMS = {
    "mlp-xs": (8, 512),
    "mlp-small": (16, 8192),
    "mlp-medium": (64, 16384),
    "mlp-large": (128, 32768),
}


class MLPGridSearch(GridSearch):
    def __init__(
        self,
        model_params,
        device_throughput,
        dram_bandwidth,
        kernel_launch_overhead,
        network_bandwidth,
        backend,
        output_file,
    ):
        super().__init__(
            model_params,
            device_throughput,
            dram_bandwidth,
            kernel_launch_overhead,
            network_bandwidth,
            backend,
            output_file,
        )

    def _get_inputs(self, batch_size, dim, num_layers):
        x = np.random.normal(size=(batch_size, dim), dtype=np.float32)
        z = np.random.normal(size=(batch_size, dim), dtype=np.float32)
        weights = [np.random.normal(size=(dim, dim), dtype=np.float32)]
        for i in range(1, num_layers - 1):
            weights.append(np.random.normal(size=(dim, dim), dtype=np.float32))
        weights.append(np.random.normal(size=(dim, dim), dtype=np.float32))
        return [x, z] + weights

    def prepare_models_and_input_data(self, topology, all_batch_sizes, all_model_sizes):
        max_batch_size = max(all_batch_sizes)
        max_num_layers = max(
            self.model_params[model_size][0] for model_size in all_model_sizes
        )
        max_dim = max(
            self.model_params[model_size][1] for model_size in all_model_sizes
        )
        if self.backend == "pytorch":
            all_input_data = self._get_inputs(
                max_batch_size, max_dim, max_num_layers, topology.devices[0]
            )
        self.models_and_input_data = {}
        for batch_size, model_size in product(all_batch_sizes, all_model_sizes):
            num_layers, dim = self.model_params[model_size]
            fn = mlp.mlp(batch_size, dim, dim, dim, num_layers, topology.devices[0])
            if self.backend == "pytorch":
                input_data = [
                    ConcreteValue(all_input_data[0][:batch_size], topology.devices[0]),
                    ConcreteValue(
                        self.all_input_data[1][:batch_size], topology.devices[0]
                    ),
                ]
                +[
                    ConcreteValue(
                        self.all_input_data[i][:dim, :dim], topology.devices[0]
                    )
                    for i in range(num_layers)
                ]
            else:
                input_data = fn.inputs
            self.models_and_input_data[(batch_size, model_size)] = (fn, input_data)

    def select_model_and_input_data(self, batch_size, model_size):
        return self.models_and_input_data[(batch_size, model_size)]

    def verify_config(
        self, batch_size, dp_degree, hp_degree, pp_degree, num_microbatches, model_size
    ):
        pass

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
        init_fn, transformed_fn = mlp_dhp_transform(
            fn,
            dp_degree,
            hp_degree,
            pp_degree,
            num_microbatches,
            topology.devices,
        )
        init_fn = infer_types(init_fn, init_fn.inputs)
        # init_function.outputs = transformed_function.inputs, so get types from there:
        transformed_fn = infer_types(transformed_fn, init_fn.outputs)
        if self.backend == "pytorch" and len(topology.devices) > 1:
            ex = SequentialExecutor("numpy")
            input_data = ex.compute(init_fn, input_data)
        else:
            input_data = transformed_fn.inputs

        return init_fn, transformed_fn, input_data

    def simulate(self, transformed_fn, input_data, topology):
        input_types = (v.type for v in input_data)
        return mlp.simulate(transformed_fn, input_types, topology)

    def pytorch(self, transformed_fn, input_data, world_size):
        return mlp.run_pytorch(transformed_fn, input_data, world_size)


def main(args):
    grid_search = MLPGridSearch(
        MODEL_PARAMS,
        args.device_throughput,
        args.dram_bandwidth,
        args.kernel_launch_overhead,
        args.network_bandwidth,
        args.backend,
        args.output_file,
    )
    grid_search.grid_search(
        args.all_world_sizes, args.all_batch_sizes, args.all_model_sizes
    )


if __name__ == "__main__":
    defaults = {
        "all_world_sizes": [1, 2, 4],
        "all_batch_sizes": [2 ** i for i in range(16)],
        "all_model_sizes": ["mlp-small", "mlp-medium", "mlp-large"],
    }
    parser = Parser(description="MLP Grid Search")
    parser.add_simulation_topology_config_arguments()
    parser.add_execution_mode_config_arguments()
    parser.add_grid_search_config_arguments(defaults)
    args = parser.parse_args()
    main(args)
