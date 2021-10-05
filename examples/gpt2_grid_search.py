import copy

from .grid_search import DHPConfig, GridSearch, run_grid_search
from . import gpt2
from .parser import Parser
from dist_ir.transforms.gpt2_dhp_transform import check_params


class GPTGridSearch(GridSearch):
    def __init__(
        self,
        backend,
        dtype,
        use_gpu,
        output_file,
        device_throughput,
        dram_bandwidth,
        kernel_launch_overhead,
        network_bandwidth,
        allreduce_parameters,
        max_world_size,
        model_path,
    ):
        super().__init__(
            gpt2.model_params,
            backend,
            dtype,
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
        self.base_model, self.base_input_data = gpt2.import_function_and_get_input_data(
            self.model_path,
            self.topology.devices[0],
            self.dtype,
            use_real_weights=(self.backend == "pytorch"),
        )
        self.models_and_input_data = {}
        self.all_input_ids = []

    def get_model_and_input_data(self, batch_size, model_size):
        pytorch_backend = self.backend == "pytorch"
        if len(self.all_input_ids) < batch_size and pytorch_backend:
            self.all_input_ids = gpt2.create_input_ids(
                batch_size, pytorch_backend
            )

        if model_size not in self.models_and_input_data:
            n_layer, n_head, d_embd = self.model_params[model_size]
            self.models_and_input_data[
                model_size
            ] = gpt2.resize_function_and_input_data(
                self.base_model,
                copy.deepcopy(self.base_input_data),
                n_layer,
                n_head,
                d_embd,
                self.dtype,
            )

        model, input_data = self.models_and_input_data[model_size]
        if pytorch_backend:
            input_ids = self.all_input_ids[:batch_size]
        else:
            input_ids = gpt2.create_input_ids(batch_size, pytorch_backend)
        input_data = [input_ids] + input_data
        return model, input_data

    def verify_config(self, config: DHPConfig):
        _, n_head, d_embd = self.model_params[config.model_size]
        check_params(
            config.batch_size,
            config.dp_degree,
            config.hp_degree,
            config.pp_degree,
            config.num_microbatches,
            n_head,
            d_embd,
        )

    def transform(
        self,
        fn,
        input_data,
        topology,
        config: DHPConfig,
    ):
        _, n_head, d_embd = self.model_params[config.model_size]
        return gpt2.transform(
            fn,
            input_data,
            topology,
            config.dp_degree,
            config.hp_degree,
            config.pp_degree,
            config.num_microbatches,
            d_embd,
            n_head,
            skip_allgathers=True,
            use_real_weights=(self.backend == "pytorch"),
        )

    def simulate(self, transformed_fn, input_data, topology):
        return gpt2.simulate(
            transformed_fn, input_data, topology, self.allreduce_parameters
        )

    def pytorch(self, transformed_fn, input_data, world_size):
        # TODO: Get num_warmup and num_repetitions from args
        return gpt2.run_pytorch(
            transformed_fn,
            input_data,
            world_size,
            num_warmup=5,
            num_repetitions=10,
            use_gpu=self.use_gpu,
        )


if __name__ == "__main__":
    defaults = {
        "all_world_sizes": [4, 8, 16],
        "all_batch_sizes": [64, 256],
        "all_model_sizes": [
            "gpt3",
            "gpt3-medium",
            "gpt3-large",
            "gpt3-xl",
            "gpt3-2.7B",
            "gpt3-6.7B",
            "gpt3-13B",
        ],
    }
    parser = Parser(description="GPT2 Grid Search")
    parser.add_simulation_config_arguments()
    parser.add_execution_mode_config_arguments()
    parser.add_grid_search_config_arguments(defaults)
    parser.add_backend_config_arguments()
    parser.add_gpt2_model_path_config_arguments()
    parser.add_model_config_arguments(choices=list(gpt2.model_params.keys()))
    args = parser.parse_args()
    run_grid_search(args, GPTGridSearch)
