from beartype import beartype
import torch.nn as nn
import torch

class Regression(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(args.n_hidden, args.n_hidden, bias=False),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(args.n_hidden, 1, bias=False)).to(args.device)

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.layers(x)
        z = z.squeeze(-1).squeeze(-1)
        return z
