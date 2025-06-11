import torch.nn as nn
import torch

class Embedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Linear(args.n_input, args.n_embed,
                                   bias=False, device=args.device)

    def forward(self, x):
        z = self.embedding(x)
        return z
