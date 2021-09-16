#! /bin/bash

# A script to run all MLSys experiments

# TODO parse args etc
machine_name=$1
world_size=$2


print_header () {
    printf '=%.0s' {1..100}
    echo $1
    printf '=%.0s' {1..100}
    echo
}


params_file="${machine_name}.params.json"
backend_file="${model_size}_backend.csv"
simulated_file="${model_size}_simulated.csv"
best_file="${model_size}_best.csv"
sample_file="${model_size}_sample.csv"

# 1. Run calibration on hardware

print_header "Calibrating simulator"
python -m examples.mlsys_experiments --mode calibrate \
    --calibrate_device_parameters --calibrate_allreduce_parameters \
    --calibrate_network_bandwidth \
    --output_file $params_file

## TODO For each model:

model="gpt"
model_size="gpt3-6.7B"

# 2. Run pure baselines on hardware

print_header "Running pure baselines"
for strategy in DP HP PP; do
    ./examples/run_pure_baseline.sh $model $model_size $strategy $world_size \
        $backend_file

# TODO
# mlsys_experiments.py reads a JSON containing BS etc and runs sim(?)
# also outputs list of inputs to pure baseline runner above
# prep-best and prep-sample should be updated to work for file containing multiple models


# 3. Run grid search using simulation to find estimated best strategies

python -m examples.gpt2_grid_search --model_path gpt2-10.onnx  \
    --backend simulate --simulation_parameters_file $params_file \
    --mode grid --all_world_sizes 1 2 4 --all_batch_sizes 512 1024 2048 4096 8192 16384 32768 --all_model_sizes $model_size \
    --output_file $simulated_file

# 4. Run best strategies on hardware

    python -m examples.mlsys_experiments --mode prep-best --simulation_file gpt_6.7B_simulated.csv --output_file gpt_6.7B_best.csv

    ./examples/run_grid_search.sh gpt gpt_6.7B_best.csv $backend_file

# 5. Run (small/random subset of) grid search on hardware for simulator accuracy

    python -m examples.mlsys_experiments --mode prep-sample --simulation_file gpt_6.7B_simulated.csv --output_file gpt_6.7B_sample.csv

    ./examples/run_grid_search.sh gpt gpt_6.7B_sample.csv $backend_file
