import torch.nn as nn
import torch

from . import module

class Actor(nn.Module):

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args
        # init model
        self.embedding = module.Embedding(args)
        self.glimpse   = module.Glimpse(args)
        self.encoder   = module.Encoder(args)
        self.decoder   = module.Decoder(args)
        self.pointer   = module.Pointer(args)
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            nn.init.uniform_(p.data, -0.08, 0.08)

    def get_log_likelihood(self, log_probs, nodes):
        # log_probs: (bs, n_node, n_node)
        # nodes: (bs, n_node)
        # return: (bs)
        ll = torch.gather(log_probs, dim=2, index=nodes[:, :, None])
        ll = torch.sum(ll.squeeze(-1), 1)
        return ll

    def forward(self, x):
        # extract parameters
        args = self.args
        bs, n_node, _ = x.size()
        # init
        nodes, log_probs = [], []
        mask = torch.zeros([bs, n_node], device=args.device)
        x = x.to(args.device)
        embedding, glimpse, encoder, decoder, pointer = \
                self.embedding, self.glimpse, self.encoder, self.decoder, self.pointer
        # embed
        e = embedding(x)
        # encode
        ref, (h, c) = encoder(e)
        # decode loop
        z = decoder.get_z0(x)
        for i in range(n_node):
            # decode
            _, (h, c) = decoder(z, h, c)
            q = h.squeeze(0)
            # glimpse
            for i in range(args.n_glimpse):
                q = glimpse(q, ref, mask)
            # pointer
            logits   = pointer(q, ref, mask)
            log_prob = torch.log_softmax(logits, dim=-1)
            # select next node
            next_node = torch.argmax(log_prob, dim=1).long() # (bs, )
            z = decoder.gather_z(e, next_node)
            # store decoding results
            nodes.append(next_node)
            log_probs.append(log_prob)
            # update mask
            mask += torch.zeros((bs, n_node), device=args.device).\
                    scatter_(dim=1, index=next_node.unsqueeze(1), value=1)
        log_probs = torch.stack(log_probs, dim=1)
        nodes = torch.stack(nodes, dim=1)
        log_likelihoods = self.get_log_likelihood(log_probs, nodes)
        return log_likelihoods, nodes
