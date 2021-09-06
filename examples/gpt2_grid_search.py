import copy

from .grid_search import GridSearch
from . import gpt2
from .parser import Parser
from dist_ir.transforms.gpt2_dhp_transform import check_params


class GPTGridSearch(GridSearch):
    def __init__(
        self,
        backend,
        use_gpu,
        output_file,
        model_path,
        device_throughput,
        dram_bandwidth,
        kernel_launch_overhead,
        network_bandwidth,
    ):
        model_params = {
            "gpt2": (12, 12, 768),
            "gpt2-medium": (24, 16, 1024),
            "gpt2-large": (36, 20, 1280),
            "gpt2-xl": (48, 25, 1600),
            "gpt2-xl": (48, 25, 1600),
            "gpt3": (12, 12, 768),
            "gpt3-medium": (24, 16, 1024),
            "gpt3-large": (24, 16, 1536),
            "gpt3-xl": (24, 16, 2048),
            "gpt3-2.7B": (32, 32, 2560),
            "gpt3-6.7B": (32, 32, 4096),
            "gpt3-13B": (40, 40, 5120),
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
        self.model_path = model_path

    def prepare_models_and_input_data(self, topology, all_batch_sizes, all_model_sizes):
        base_model, base_input_data = gpt2.import_function_and_get_input_data(
            self.model_path,
            topology.devices[0],
            use_real_weights=(self.backend == "pytorch"),
        )
        self.models_and_input_data = {}
        for model_size in all_model_sizes:
            n_layer, n_head, d_embd = self.model_params[model_size]
            self.models_and_input_data[
                model_size
            ] = gpt2.resize_function_and_input_data(
                base_model,
                copy.deepcopy(base_input_data),
                n_layer,
                n_head,
                d_embd,
            )
        self.all_input_ids = gpt2.create_input_ids(max(all_batch_sizes))

    def select_model_and_input_data(self, batch_size, model_size):
        model, input_data = self.models_and_input_data[model_size]
        input_ids = self.all_input_ids[:batch_size]
        input_data = [input_ids] + input_data
        return model, input_data

    def verify_config(
        self, batch_size, dp_degree, hp_degree, pp_degree, num_microbatches, model_size
    ):
        n_layer, n_head, d_embd = self.model_params[model_size]
        check_params(
            batch_size,
            dp_degree,
            hp_degree,
            pp_degree,
            num_microbatches,
            n_head,
            d_embd,
        )

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
        n_layer, n_head, d_embd = self.model_params[model_size]
        return gpt2.transform(
            fn,
            input_data,
            topology,
            dp_degree,
            hp_degree,
            pp_degree,
            num_microbatches,
            d_embd,
            n_head,
            use_real_weights=(self.backend == "pytorch"),
        )

    def simulate(self, transformed_fn, input_data, topology):
        return gpt2.simulate(transformed_fn, input_data, topology)

    def pytorch(self, transformed_fn, input_data, world_size):
        return gpt2.run_pytorch(
            transformed_fn, input_data, world_size, use_gpu=self.use_gpu
        )


def main(args):
    grid_search = GPTGridSearch(
        args.backend,
        args.use_gpu,
        args.output_file,
        args.model_path,
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
    parser.add_simulation_topology_config_arguments()
    parser.add_execution_mode_config_arguments()
    parser.add_grid_search_config_arguments(defaults)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to GPT-2 ONNX model "
            "(downloaded from https://github.com/onnx/models/blob/master/"
            "text/machine_comprehension/gpt-2/model/gpt2-10.onnx?raw=True)"
        ),
    )
    args = parser.parse_args()
    main(args)
