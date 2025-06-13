from torch.nn.utils.rnn import PackedSequence
from beartype import beartype
import torch.nn as nn
import torch

class Decoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layer = nn.LSTM(input_size=args.n_embed,
                             hidden_size=args.n_hidden,
                             batch_first=True, 
                             device=args.device)
        self.g = nn.Parameter(torch.FloatTensor(args.n_embed).to(args.device))

    @beartype
    def get_z0(self, x: PackedSequence) -> torch.Tensor:
        bs = x.unsorted_indices.numel()
        args = self.args
        z0 = self.g.unsqueeze(0).repeat(bs, 1).unsqueeze(1).to(args.device)
        return z0

    @beartype
    def gather_z(self, e: PackedSequence, next_node: torch.Tensor) -> torch.Tensor:
        args = self.args
        # e: (bs, n_node, n_embed)
        padded_e = torch.nn.utils.rnn.pad_packed_sequence(e)[0].permute(1, 0, 2)
        index = next_node.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, args.n_embed)
        z = torch.gather(padded_e, dim=1, index=index)
        return z
    
    @beartype
    def forward(self, z: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        z, (h, c) = self.layer(z, (h, c))
        return z, (h, c)
