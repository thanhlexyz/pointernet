import torch.nn.functional as F
import torch.nn as nn
import torch

from . import module

class Actor(nn.Module):

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args
        # init model
        self.embedding  = module.Embedding(args)
        self.glimpse    = module.Glimpse(args)
        self.encoder    = module.Encoder(args)
        self.regression = module.Regression(args)
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            nn.init.uniform_(p.data, -0.08, 0.08)

    def forward(self, x):
        # extract parameters
        args = self.args
        bs, n_node, _ = x.size()
        # init
        nodes, log_probs = [], []
        mask = torch.zeros([bs, n_node], device=args.device)
        x = x.to(args.device)
        # embed
        e = embedding(x)
        # encode
        ref, (h, c) = encoder(e)
        q = h[-1]
        for _ in range(args.n_process):
            for _ in range(args.n_glimpse):
				query = self.glimpse(query, ref)
        # predict
        value = self.regression(query)
        return value
