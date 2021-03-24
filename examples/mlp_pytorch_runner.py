import argparse
import subprocess
from multiprocessing.pool import ThreadPool
import re
import os

def run(config):
    args, i = config
    cd = os.path.dirname(os.path.realpath(__file__))
    command = (f"python {cd}/mlp_pytorch.py --world_size {args.world_size} "
            f"--rank {i} --num_warmup_steps {args.num_warmup_steps} ")
            #f"--num_profiling_steps {args.num_profiling_steps "
            #f"--batch_size {args.batch_size} --hidden_dim {args.hidden_dim} "
            #"--num_hideen_layers {args.num_hidden_layers")
    ret = subprocess.run(command, capture_output=True, shell=True)
    ret.check_returncode()
    return ret.stdout

def main(args):
    configs = [(args, i) for i in range(args.world_size)] 
    with ThreadPool(args.world_size) as p:
        results = p.map(run, configs)
    for result in results:
        if len(result) > 0:
            runtime = float(re.match("Median runtime: ([-+]?[0-9]*\.?[0-9]+) ms", result).group(1))
            break
    print("World size {args.world_size}: median runtime = {runtime} ms")

if __name__=='__main__':
    parser = argparse.ArgumentParser("Runner for PyTorch MLP experiments")
    parser.add_argument("--world_size", type=int, required=True, help="World size")
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="# profiling steps")
    parser.add_argument("--num_profiling_steps", type=int, default=1000, help="# profiling steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dim")
    parser.add_argument(
        "--num_hidden_layers", type=int, default=8, help="# hidden layers"
    )
    args = parser.parse_args()
    main(args)
