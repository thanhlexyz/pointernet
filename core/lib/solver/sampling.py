import numpy as np
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
        # create actor/critic model
        self.create_model()
        # load monitor
        self.monitor = lib.Monitor(args)
        # n_test
        self.n_test = args.n_test_instance

    def test_item(self, item):
        # extract args
        args = self.args
        # extract model
        actor = self.actor
        # init
        l_best = torch.inf
        # extract data
        x, _ = item.values()
        # x = torch.tensor(x, device=args.device)
        x = x.repeat(args.batch_size, 1, 1).to(args.device)
        x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
        
        # get actor prediction
        for _ in range(args.n_sample_step):
            _, y_hat = actor(x)
            l = util.get_tour_length(x, y_hat)
            idx = torch.argmin(l)
            l = l[idx]
            y_hat = y_hat[idx]
            if l < l_best:
                l_best = l.item()
        # gather info
        return l_best


    def test(self):
        # extract args
        args = self.args
        monitor = self.monitor
        monitor.create_progress_bar(self.n_test)
        args.n_logging = 1
        self.load_model()
        self.actor.eval()
        ls = []
        for l in self.test_epoch():
            ls.append(l)
            info = {'avg_tour_length': np.mean(ls)}
            monitor.step(info)
        monitor.export_csv(mode='test')
