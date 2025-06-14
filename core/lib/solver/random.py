import torch
import math

import lib.util as util
import lib
from .base import SolverBase

class Solver(SolverBase):

    def __init__(self, args):
        # save args
        self.args = args
        # create dataloader dict
        self.dataloader_dict = lib.dataset.create(args)
        # load monitor
        self.monitor = lib.Monitor(args)
        # n_test
        self.n_test = math.ceil(args.n_test_instance / args.batch_size)

    def test_epoch(self):
        # extract args
        args = self.args
        dataloader = self.dataloader_dict['test']
        # training loop
        for x, _ in dataloader:
            # extract data
            x = x.to(args.device)
            bs = x.unsorted_indices.numel()
            y_hat = []
            _, N = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            for i in range(bs):
                y_hat.append(torch.randperm(N[i]))
            y_hat = torch.nn.utils.rnn.pack_sequence(y_hat, enforce_sorted=False)
            l = util.get_tour_length(x, y_hat)
            # gather info
            yield l.detach().cpu().numpy().tolist()

