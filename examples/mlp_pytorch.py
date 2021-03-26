import argparse
import numpy as np
import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

class Mlp(torch.nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim):
        super().__init__()
        self.blocks = [
            torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_hidden_layers)
        ]
        for i, block in enumerate(self.blocks):
            for j, param in enumerate(block.parameters()):
                if j > 0:
                    raise ValueError(f"Block {i} has more than 1 parameter!")
                self.register_parameter(f"w{chr(ord('A')+i)}", param)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            relu = torch.nn.ReLU()
            x = relu(x)
        return x

def setup(args):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    torch.distributed.init_process_group(args.backend, world_size=len(args.ranks), rank=args.local_rank)

def cleanup():
    torch.distributed.destroy_process_group()

def run(config):
    (args, rank) = config
    args.local_rank = rank
    torch.cuda.set_device(args.local_rank)
    setup(args)
    model = Mlp(args.num_hidden_layers, args.hidden_dim).to(args.local_rank)
    ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    loss_fn = torch.nn.MSELoss()
    #optimizer = torch.optim.SGD(ddp_model.parameters, lr=0.001)
    #optimizer.zero_grad()
    x = torch.randn(size=(args.batch_size // len(args.ranks), args.hidden_dim)).to(args.local_rank)
    labels = torch.randn(size=(args.batch_size // len(args.ranks), args.hidden_dim)).to(args.local_rank)
    runtimes = []
    for i in range(args.num_warmup_steps + args.num_profiling_steps):
        if i % 10 == 0:
            print(f"rank {args.local_rank}: iter {i}")
        start = time.time()
        y = model(x)
        loss_fn(y, labels).backward()
        duration = time.time() - start
        runtimes.append(duration)
    #optimizer.step()
    cleanup()

    return np.median(runtimes[args.num_warmup_steps:])
    #print(np.median(runtimes[args.num_warmup_steps:]))

def main(args):
    configs = [(args, rank) for rank in args.ranks]
    with torch.multiprocessing.Pool(len(args.ranks)) as p:
        results = p.map(run, configs)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch distributed MLP")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master address")
    parser.add_argument("--master_port", type=str, default="12355", help="Master port") 
    parser.add_argument("--ranks", nargs="+", type=int, help="Ranks")
    parser.add_argument("--backend", type=str, default="nccl", help="Backend")
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="# profiling steps")
    parser.add_argument("--num_profiling_steps", type=int, default=1000, help="# profiling steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dim")
    parser.add_argument(
        "--num_hidden_layers", type=int, default=8, help="# hidden layers"
    )
    args = parser.parse_args()
    main(args)
