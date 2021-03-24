import argparse
import torch


class Mlp(torch.nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim):
        super().__init__()
        self.blocks = [
            torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_hidden_layers)
        ]

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            relu = torch.nn.ReLU()
            x = relu(x)
        return x


def main(args):
    model = Mlp(args.num_hidden_layers, args.hidden_dim)
    x = torch.randn(size=(args.batch_size, args.hidden_dim))
    y = model(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch distributed MLP")
    parser.add_argument("--world_size", type=int, help="World size")
    parser.add_argument("--rank", type=int, help="Rank")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dim")
    parser.add_argument(
        "--num_hidden_layers", type=int, default=8, help="# hidden layers"
    )
    args = parser.parse_args()
    main(args)
