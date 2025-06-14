from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from beartype import beartype
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle
import torch
import tqdm
import os

import lib.util as util
import lib

from .base import SolverBase

class Solver(SolverBase):

    def __init__(self, args):
        # save args
        self.args = args
        # create dataloader dict
        self.dataloader_dict = lib.dataset.create(args)
        # create actor/critic model
        self.create_model()
        # load monitor
        self.monitor = lib.Monitor(args)
        # n_sample_step
        self.n_test = args.n_test_instance

    @beartype
    def shuffle(self, x: PackedSequence) -> PackedSequence:
        x_padded, lengths = pad_packed_sequence(x, batch_first=True)
        args = self.args
        x_shuffle = torch.zeros_like(x_padded)
        for i in range(args.batch_size):
            x_shuffle[i] = x_padded[i, torch.randperm(len(x_padded[i]))]
        return pack_padded_sequence(x_shuffle, lengths, batch_first=True, enforce_sorted=False)

    def test_item(self, item):
        # extract args
        args = self.args
        # extract model
        self.load_model()
        self.actor.eval()
        actor_optimizer = self.actor_optimizer
        actor = self.actor
        actor.train()
        # init
        l_best = torch.inf
        # extract data
        x, y = item.values()
        # replicate x for batch processing:
        x = x.repeat(args.batch_size, 1, 1).to(args.device)
        x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
        
        # get baseline tour length from a random tour
        y_rand = torch.stack([torch.randperm(len(y)) for i in range(args.batch_size)], dim=0)
        y_rand = torch.nn.utils.rnn.pack_sequence(y_rand, enforce_sorted=False)
        
        b = util.get_tour_length(x, y_rand).squeeze()

        for _ in range(args.n_sample_step):
            # shuffle node order to improve diversity
            x = self.shuffle(x)
            # get actor prediction
            log_likelihood, y_hat = actor(x)
            l = util.get_tour_length(x, y_hat)
            # track l_best (smallest tour length)
            idx = torch.argmin(l)
            if l[idx] < l_best:
                l_best = l[idx].item()
            # update actor
            advantage = l - b
            actor_optimizer.zero_grad()
            actor_loss = torch.mean(advantage * log_likelihood)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1., norm_type=2)
            actor_optimizer.step()
            b = b * args.active_search_alpha + (1 - args.active_search_alpha) * torch.mean(l, dim=0)
        # gather info
        return l_best
