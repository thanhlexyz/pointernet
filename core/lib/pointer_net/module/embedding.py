from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
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
        padded_x, lengths = pad_packed_sequence(x, batch_first=True)
        z = self.embedding(padded_x)
        return pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
