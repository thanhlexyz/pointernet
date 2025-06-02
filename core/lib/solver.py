from torch.utils.data import DataLoader
import numpy as np
import torch
import time
import os

import lib


class Solver:

    def __init__(self, args):
        # save args
        self.args = args
        # create dataset and dataloader
        self.dataset = lib.create_dataset(args)
        self.dataset.prepare()
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True if args.mode == 'train' else False,
                                     num_workers=os.cpu_count())
        # create net
        self.net = lib.create_net(args)
        # load monitor
        self.monitor = lib.Monitor(args)
    # =========================================
    def train_add_info(self):
        pass

    def train_epoch(self):
        pass

    def train(self):
        pass
    # =========================================
    def test_add_info(self):
        pass

    def test(self):
        # extract args
        dataloader = self.dataloader
        monitor    = self.monitor
        args       = self.args
        net        = self.net
        # evaluate each batch
        for i, batch in enumerate(dataloader):
            x, y    = batch.values()
            print(x.dtype)
            exit()
            y_hat   = net(x)
            l_opt   = get_tour_length(x, y)
            l       = get_tour_length(x, y_hat)
            opt_gap = l - l_opt
            print(opt_gap)
            exit()
