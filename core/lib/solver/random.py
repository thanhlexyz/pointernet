from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle
import torch
import time
import tqdm
import os

import lib.util as util
import lib


class Solver:

    def __init__(self, args):
        # save args
        self.args = args
        # create dataloader dict
        self.dataloader_dict = lib.dataset.create(args)
        # load monitor
        self.monitor = lib.Monitor(args)

    def test_epoch(self):
        # extract args
        args = self.args
        dataloader = self.dataloader_dict['test']
        # training loop
        for batch in dataloader:
            # extract data
            x, _ = batch.values()
            x = x.to(args.device)
            y_hat = torch.randperm(args.n_node)
            l = util.get_tour_length(x, y_hat)
            # gather info
            yield l

    def test(self):
        # extract args
        args = self.args
        monitor = self.monitor
        ls = []
        for l in self.test_epoch():
            ls.append(l)
        ls = torch.cat(ls).detach().cpu().numpy()
        info = {'avg_tour_length': np.mean(ls)}
        monitor.step(info)
        monitor.export_csv(mode='test')
