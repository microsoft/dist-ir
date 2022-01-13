#! /bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# A script to run grid search using the pytorch backend.
# Given a file of configurations, we run them one by one and do clean-up etc in
# between.

if [[ $# -lt 4 || ! ( "$1" =~ ^(mlp|gpt)$ ) ]]; then
    echo "Usage: $0 <mlp|gpt> <world size> <configs file> <output file>"
    echo "Runs grid search using pytorch backend on all configs in <configs file>"
    exit 1
fi
model=$1
world_size=$2
configs_file=$3
output_file=$4

if [[ "$model" == "mlp" ]]; then
    module="mlp_grid_search"
    model_path_arg=""
else
    module="gpt2_grid_search"
    model_path_arg="--model_path gpt2-10.onnx"
fi

num_configs=`wc -l < $configs_file`
for ((i=1;i<$num_configs;i++)); do
    command="examples.$module $model_path_arg --mode file --backend pytorch --use_gpu \
        --configs_file $configs_file --config_number $i \
        --output_file $output_file --append_output_file \
        --all_world_sizes $world_size \
        "
    timeout 45m python -m $command
    retcode=$?
    if [[ $retcode == 124 ]]; then
        echo "TIMEOUT"
        echo
    fi
done
