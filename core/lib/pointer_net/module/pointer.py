from torch.nn.utils.rnn import PackedSequence
from beartype import beartype
import torch.nn as nn
import torch

class Pointer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.v = nn.Parameter(torch.FloatTensor(args.n_hidden).to(args.device))
        self.W_ref = nn.Conv1d(args.n_hidden, args.n_hidden, 1, 1,
                               device=args.device)
        self.W_q = nn.Linear(args.n_hidden, args.n_hidden,
                             bias=True, device=args.device)

    @beartype
    def forward(self, q: torch.Tensor, ref: PackedSequence, mask: torch.Tensor, inf: float = 1e8) -> torch.Tensor:
        args = self.args
        # ref: (bs, n_node, n_hidden)
        ref = torch.nn.utils.rnn.pad_packed_sequence(ref)[0].permute(1, 0, 2)
        # u1: (bs, n_hidden, n_node)
        u1 = self.W_q(q).unsqueeze(-1).repeat(1, 1, ref.size(1))
        # u2: (bs, n_hidden, n_node)
        u2 = self.W_ref(ref.permute(0, 2, 1))
        V = self.v.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        # u: (bs, n_node)
        u = torch.bmm(V, args.clip_logit * torch.tanh(u1 + u2)).squeeze(1)
        u = u - inf * mask
        return u
