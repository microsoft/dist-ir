#! /bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# A script to run pure strategy baslines using the pytorch backend.
# Given the strategy DP/HP/PP, we run with increasing batch_sizes until we find
# find the config with the highest throughput.

if [[ $# -lt 5 || ! ( "$1" =~ ^(mlp|gpt)$ ) || ! ( "$3" =~ ^(DP|HP|PP)$ ) ]]; then
    echo "Usage: $0 <mlp|gpt> <model size> <DP|HP|PP> <world size> <output file>"
    echo "Runs a pure strategy using pytorch backend and saves output to <output file>"
    exit 1
    # TODO use --arguments, parse robustly
fi
model=$1
model_size=$2
strategy=$3
world_size=$4
output_file=$5

if [[ "$model" == "mlp" ]]; then
    module="mlp_grid_search"
    model_path_arg=""
else
    module="gpt2_grid_search"
    model_path_arg="--model_path gpt2-10.onnx"
fi

case $strategy in
    DP)
        config="$world_size 1 1 1"
        ;;
    HP)
        config="1 $world_size 1 1"
        ;;
    PP)
        config="1 1 $world_size 32"  # TODO num_microbatches?
        ;;
    *)
        echo "Unknown strategy"
        exit 1
        ;;
esac

for ((i=10;i<20;i++)); do
    batch_size=$((2**i))
    command="examples.$module $model_path_arg --mode config --backend pytorch \
        --model_size $model_size --config $config $batch_size \
        --output_file $output_file --append_output_file \
        "
    python -m $command || exit 1
done
