import torch.nn as nn
import torch

class Glimpse(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.v = nn.Parameter(torch.FloatTensor(args.n_hidden).to(args.device))
        self.W_ref = nn.Conv1d(args.n_hidden, args.n_hidden, 1, 1,
                               device=args.device)
        self.W_q = nn.Linear(args.n_hidden, args.n_hidden,
                             bias=True, device=args.device)

    def forward(self, q, ref, mask=None, inf=1e8):
        # extract args
        args = self.args
        # u1: (bs, n_hidden, n_node)
        u1 = self.W_q(q).unsqueeze(-1).repeat(1, 1, ref.size(1))
        # u2: (bs, n_hidden, n_node)
        u2 = self.W_ref(ref.permute(0, 2, 1))
        # v: (bs, 1, n_hidden) * u1+u2: (bs, n_hidden, n_node)
        v  = self.v.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        # u: (bs, n_node)
        u = torch.bmm(v, torch.tanh(u1 + u2)).squeeze(1)
        if mask is not None:
            u = u - inf * mask
        # a: (batch, n_node, i1)
        a = F.softmax(u / args.softmax_temperature, dim = 1)
        # d: (bs, n_hidden)
        d = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
        return d
