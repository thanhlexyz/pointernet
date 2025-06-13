import torch.nn as nn
import torch
from torch.nn.utils.rnn import PackedSequence

class Encoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.layer = nn.LSTM(input_size=args.n_embed,
                             hidden_size=args.n_hidden,
                             batch_first=True, device=args.device)

    def forward(self, e: PackedSequence) -> PackedSequence:
        z, (h, c) = self.layer(e, None)
        return z, (h, c)
