#! /bin/bash

# A script to run grid search using the pytorch backend.
# Given a file of configurations, we run them one by one and do clean-up etc in
# between.

if [[ $# -lt 3 || ! ( "$1" =~ ^(mlp|gpt)$ ) ]]; then
    echo "Usage: $0 <mlp|gpt> <configs-file> <output-file>"
    echo "Runs grid search using pytorch backend on all configs in <configs-file>"
    exit 1
fi

num_configs=`wc -l < $2`
for ((i=1;i<$num_configs;i++)); do
    if [[ "$1" == "mlp" ]]; then
        python -m examples.mlp_grid_search --backend pytorch \
            --configs_file $2 --config_number $i \
            --output_file $3 --append_output_file
    else
        python -m examples.gpt2_grid_search --backend pytorch \
            --model_path gpt2-10.onnx \
            --configs_file $2 --config_number $i \
            --output_file $3 --append_output_file
    fi
done
