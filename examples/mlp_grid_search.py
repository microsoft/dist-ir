from itertools import product
import numpy as np
import argparse

from dist_ir.ir import Value
from dist_ir.ir.type import Tensor
from dist_ir.executor import infer_types, SequentialExecutor, ConcreteValue
from dist_ir.transforms import mlp_dhp_transform
from . import mlp
from .grid_search import GridSearch
from .parser import Parser


class MLPGridSearch(GridSearch):
    def __init__(
        self,
        backend,
        use_gpu,
        output_file,
        device_throughput,
        dram_bandwidth,
        kernel_launch_overhead,
        network_bandwidth,
    ):
        model_params = {
            "mlp-xs": (8, 512),
            "mlp-small": (16, 8192),
            "mlp-medium": (64, 16384),
            "mlp-large": (128, 32768),
        }
        super().__init__(
            model_params,
            backend,
            use_gpu,
            output_file,
            device_throughput,
            dram_bandwidth,
            kernel_launch_overhead,
            network_bandwidth,
        )

    def prepare_models_and_input_data(self, topology, all_batch_sizes, all_model_sizes):
        max_batch_size = max(all_batch_sizes)
        max_num_layers = max(
            self.model_params[model_size][0] for model_size in all_model_sizes
        )
        max_dim = max(
            self.model_params[model_size][1] for model_size in all_model_sizes
        )
        self.models = {}
        for model_size in all_model_sizes:
            num_layers, dim = self.model_params[model_size]
            self.models[model_size] = mlp.mlp(
                max_batch_size, dim, dim, dim, num_layers, topology.devices[0]
            )

    def get_model_and_input_data(self, batch_size, model_size):
        fn = self.models[model_size]
        num_layers, dim = self.model_params[model_size]
        if self.backend == "pytorch":
            input_data = mlp.get_input_data(batch_size, dim, num_layers)
            input_data = tuple(
                ConcreteValue(input_data[i], fn.inputs[i].type.device)
                for i in range(len(input_data))
            )
        else:
            input_data = list(fn.inputs)
            # Update x and z to use the selected batch size
            for i in range(2):
                input_data[i] = Value(
                    fn.inputs[i].name,
                    Tensor(
                        shape=(batch_size, dim),
                        dtype=input_data[i].type.dtype,
                        device=input_data[i].type.device,
                    ),
                )
            input_data = tuple(input_data)
        return fn, input_data

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
        transformed_fn = mlp.add_optimizer_ops(transformed_fn)
        if self.backend == "pytorch":
            if len(topology.devices) > 1:
                ex = SequentialExecutor("numpy")
                input_data = ex.compute(init_fn, input_data)
        else:
            input_data = transformed_fn.inputs

        return init_fn, transformed_fn, input_data

    def simulate(self, transformed_fn, input_data, topology):
        input_types = (v.type for v in input_data)
        return mlp.simulate(transformed_fn, input_types, topology)

    def pytorch(self, transformed_fn, input_data, world_size):
        return mlp.run_pytorch(
            transformed_fn, input_data, world_size, use_gpu=self.use_gpu
        )


def main(args):
    grid_search = MLPGridSearch(
        args.backend,
        args.use_gpu,
        args.output_file,
        args.device_throughput,
        args.dram_bandwidth,
        args.kernel_launch_overhead,
        args.network_bandwidth,
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
    parser.add_backend_config_arguments()
    args = parser.parse_args()
    main(args)
