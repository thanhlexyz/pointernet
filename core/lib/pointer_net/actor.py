from torch.nn.utils.rnn import PackedSequence
from beartype import beartype
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

    @beartype
    def get_log_likelihood(self, log_probs: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        # log_probs: (bs, max_node, max_node)
        # nodes: (bs, max_node)
        # return: (bs)
        ll = torch.gather(log_probs, dim=2, index=nodes[:, :, None])
        ll = torch.sum(ll.squeeze(-1), 1)
        return ll

    @beartype
    def forward(self, x: PackedSequence) -> tuple[torch.Tensor, PackedSequence]:
        # extract parameters
        args = self.args
        bs = x.unsorted_indices.numel()
        max_node = len(x.batch_sizes)
        # init
        nodes, log_probs = [], []
        mask = torch.zeros([bs, max_node], device=args.device)

        x = x.to(args.device)
        # embed
        e = self.embedding(x)
        # encode
        ref, (h, c) = self.encoder(e)
        # decode loop
        z = self.decoder.get_z0(x)

        # get actual number of nodes
        _, n_nodes = torch.nn.utils.rnn.pad_packed_sequence(x)
        
        # fill mask with 1 for padding
        for b in range(bs):
            mask[b, n_nodes[b]:] = 1.0

        for _ in range(max_node):
            # decode
            _, (h, c) = self.decoder(z, h, c)
            q = h.squeeze(0)
            # glimpse
            for _ in range(args.n_glimpse):
                q = self.glimpse(q, ref, mask)
            # pointer
            logits   = self.pointer(q, ref, mask)
            log_prob = torch.log_softmax(logits, dim=-1)
            # select next node
            next_node = torch.argmax(log_prob, dim=1).long()
            z = self.decoder.gather_z(e, next_node)
            # store decoding results
            nodes.append(next_node)
            log_probs.append(log_prob)
            # update mask
            mask += torch.zeros((bs, max_node), device=args.device).\
                    scatter_(dim=1, index=next_node.unsqueeze(1), value=1)
        log_probs = torch.stack(log_probs, dim=1)
        # stack padded nodes
        nodes = torch.stack(nodes, dim=1)
        log_likelihoods = self.get_log_likelihood(log_probs, nodes)

        # pack nodes
        nodes = [node[:n_nodes[i]] for i, node in enumerate(nodes)]
        nodes = torch.nn.utils.rnn.pack_sequence(nodes, enforce_sorted=False)
        return log_likelihoods, nodes
