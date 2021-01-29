from ..ir.function import FunctionMaker
from ..ir.op import Op
from ..ir import Device, cpprint
from .shard_transform import shard_transform
from .pipeline_parallel_transform import PipelineParallelTransform


def hybrid_transform(function, dp_config, hp_config, pp_config=None):
    # Apply data parallel transform.
    function = shard_transform(
        function,
        function.ops,
        dp_config["input_dims"],
        dp_config["reduction_params"],
        dp_config["devices"],
        dp_config["verify_fn"],
    )
    cpprint(function)
    # Get a mutable representation of the function.
    function = FunctionMaker(
        function.name, list(function.ops), list(function.inputs), list(function.outputs)
    )
    for i, dp_op in enumerate(function.ops):
        if dp_op.op_type == "Pmap":
            # Apply horizontal parallel transform and update the
            # data parallel Pmap subfunction.
            subfunction = shard_transform(
                dp_op.subfunctions[0],
                dp_op.subfunctions[0].ops,
                hp_config["input_dims"],
                hp_config["reduction_params"],
                hp_config["devices"],
                hp_config["verify_fn"],
            )
            cpprint(subfunction)
            # TODO: Is this necessary here?
            function.ops[i] = Op(
                op_type="Pmap",
                inputs=dp_op.inputs,
                attributes=dp_op.attributes,
                subfunctions=(subfunction,),
                output_values=dp_op.outputs,
            )
            if pp_config is None:
                return function.finalize()
            else:
                subfunction = FunctionMaker(
                    subfunction.name,
                    list(subfunction.ops),
                    list(subfunction.inputs),
                    list(subfunction.outputs),
                )

            for j, hp_op in enumerate(subfunction.ops):
                if hp_op.op_type == "Pmap":
                    # Apply pipeline parallel transform and update the
                    # horizontal parallel Pmap subfunction.
                    pipeline_parallel_transform = PipelineParallelTransform(
                        pp_config["num_microbatches"],
                        pp_config["batch_dims"],
                        pp_config["reduction_params"],
                        pp_config["partition_map"],
                        pp_config["schedule"],
                    )
                    subsubfunction = pipeline_parallel_transform.apply(
                        hp_op.subfunctions[0]
                    )
                    subfunction.ops[j] = Op(
                        op_type="Pmap",
                        name=hp_op.name,
                        inputs=hp_op.inputs,
                        attributes=hp_op.attributes,
                        subfunctions=(subsubfunction,),
                        output_values=hp_op.outputs,
                    )
                    subfunction = subfunction.finalize()
                    function.ops[i] = Op(
                        op_type="Pmap",
                        inputs=dp_op.inputs,
                        attributes=dp_op.attributes,
                        subfunctions=(subfunction,),
                        output_values=dp_op.outputs,
                    )
                    return function.finalize()
