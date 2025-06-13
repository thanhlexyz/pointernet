import torch.nn as nn
import torch
from torch.nn.utils.rnn import PackedSequence

class Embedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Linear(args.n_input, args.n_embed,
                                   bias=False, device=args.device)

    def forward(self, x: PackedSequence) -> PackedSequence:
        z = self.embedding(x.data)
        return PackedSequence(z, x.batch_sizes)
