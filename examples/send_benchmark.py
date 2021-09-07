import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


NUM_WARMUP_TRIALS = 25
NUM_TRIALS = 10


def send(rank, src_rank, world_size, group_ranks):
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # TODO make these configurable
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.new_group(group_ranks)
    runtimes = []
    x = torch.randn(size=(8192, 8192), dtype=torch.float32).to(f"cuda:{rank}")
    for i in range(NUM_WARMUP_TRIALS + NUM_TRIALS):
        # torch.distributed.barrier(group=group)
        start = time.time()
        dist.broadcast(x, src_rank, group=group)
        torch.cuda.synchronize(device=rank)
        end = time.time()
        runtimes.append(end - start)
    dist.destroy_process_group()
    print(f"Send latencies: {runtimes[NUM_WARMUP_TRIALS:]}")


def recv(rank, src_rank, world_size, group_ranks):
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # TODO make these configurable
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.new_group(group_ranks)
    runtimes = []
    x = torch.zeros(size=(8192, 8192), dtype=torch.float32).to(f"cuda:{rank}")
    for i in range(NUM_WARMUP_TRIALS + NUM_TRIALS):
        # torch.distributed.barrier(group=group)
        start = time.time()
        dist.broadcast(x, src_rank, group=group)
        torch.cuda.synchronize(device=rank)
        end = time.time()
        runtimes.append(end - start)
    dist.destroy_process_group()
    print(f"Recv latencies: {runtimes[NUM_WARMUP_TRIALS:]}")


def main(args):
    p_src = mp.Process(target=send, args=(args.src_rank, args.src_rank, 2, [0, 1]))
    p_dst = mp.Process(target=recv, args=(1 - args.src_rank, args.src_rank, 2, [0, 1]))

    p_src.start()
    p_dst.start()
    p_src.join()
    p_dst.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_rank", choices=[0, 1], type=int, required=True)
    args = parser.parse_args()
    main(args)
