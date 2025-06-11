import torch.nn as nn
import torch

class Decoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layer = nn.LSTM(input_size=args.n_embed,
                             hidden_size=args.n_hidden,
                             batch_first=True, device=args.device)
        self.g = nn.Parameter(torch.FloatTensor(args.n_embed).to(args.device))

    def get_z0(self, x):
        bs, _, _ = x.size()
        args = self.args
        z0 = self.g.unsqueeze(0).repeat(bs, 1).unsqueeze(1).to(args.device)
        return z0

    def gather_z(self, e, next_node):
        args = self.args
        index = next_node.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, args.n_embed)
        z = torch.gather(e, dim=1, index=index)
        return z

    def forward(self, z, h, c):
        z, (h, c) = self.layer(z, (h, c))
        return z, (h, c)
