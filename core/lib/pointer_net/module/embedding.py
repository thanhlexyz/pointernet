from torch.nn.utils.rnn import PackedSequence
from beartype import beartype
import torch.nn as nn

class Embedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Linear(args.n_input, 
                                   args.n_embed,
                                   bias=False, 
                                   device=args.device)

    @beartype
    def forward(self, x: PackedSequence) -> PackedSequence:
        z = self.embedding(x.data)
        return PackedSequence(z, x.batch_sizes)
