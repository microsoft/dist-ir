from dist_ir.ir import Value
from dist_ir.ir.type import Tensor, abstract_values
from dist_ir.executor import infer_types, SequentialExecutor, ConcreteValue
from dist_ir.transforms import mlp_dhp_transform
from . import mlp
from .grid_search import DHPConfig, GridSearch, run_grid_search
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
        allreduce_parameters,
        max_world_size,
        model_path=None,
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
            allreduce_parameters,
            max_world_size,
            model_path,
        )
        self.models = {}

    def get_model_and_input_data(self, batch_size, model_size):
        if model_size not in self.models:
            num_layers, dim = self.model_params[model_size]
            self.models[model_size] = mlp.mlp(
                dim, dim, dim, num_layers, self.topology.devices[0]
            )

        fn = self.models[model_size]
        num_layers, dim = self.model_params[model_size]
        if self.backend == "pytorch":
            input_data = mlp.get_input_data(batch_size, dim, num_layers)
            input_data = tuple(
                ConcreteValue(t, inp.type.device)
                for t, inp in zip(input_data, fn.inputs)
            )
        else:
            input_data = mlp.get_typed_input_values(fn.inputs, batch_size, dim, dim)
        return fn, input_data

    def verify_config(self, config: DHPConfig):
        pass

    def transform(
        self,
        fn,
        input_data,
        topology,
        config: DHPConfig,
    ):
        init_fn, transformed_fn = mlp_dhp_transform(
            fn,
            config.dp_degree,
            config.hp_degree,
            config.pp_degree,
            config.num_microbatches,
            topology.devices,
        )
        init_fn = infer_types(init_fn, input_data)
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


if __name__ == "__main__":
    defaults = {
        "all_world_sizes": [1, 2, 4],
        "all_batch_sizes": [2 ** i for i in range(16)],
        "all_model_sizes": ["mlp-small", "mlp-medium", "mlp-large"],
    }
    parser = Parser(description="MLP Grid Search")
    parser.add_simulation_config_arguments()
    parser.add_execution_mode_config_arguments()
    parser.add_grid_search_config_arguments(defaults)
    parser.add_backend_config_arguments()
    args = parser.parse_args()
    run_grid_search(args, MLPGridSearch)
