from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
import time
import tqdm
import os

from . import util
import lib


class Solver:

    def __init__(self, args):
        # save args
        self.args = args
        # create dataset and dataloader
        self.train_dataset = lib.create_dataset('train', args)
        self.train_dataset.prepare()
        self.test_dataset = lib.create_dataset('test', args)
        self.test_dataset.prepare()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=os.cpu_count())
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=os.cpu_count())
        # create net
        self.net = lib.create_net(args).to(args.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                    self.net.parameters()),
                         lr=args.learning_rate)
        # load monitor
        self.monitor = lib.Monitor(args)
    # =========================================
    def train_epoch(self):
        # extract args
        loss_function = self.loss_function
        dataloader    = self.train_dataloader
        optimizer     = self.optimizer
        args          = self.args
        net           = self.net
        # evaluate each batch
        opt_gaps = []
        losses   = []
        # for batch in dataloader:
        for batch in tqdm.tqdm(dataloader):
            x, y        = batch.values()
            x = x.to(args.device)
            y = y.to(args.device)
            # x.shape = (bs, n_node, 2); y.shape = (bs, n_step)
            net(x)
            exit()
            # prob.shape (bs, step id, node id)
            prob        = prob.contiguous().view(-1, prob.size()[-1])
            loss        = loss_function(prob, y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            l_opt       = util.get_tour_length(x, y)
            l           = util.get_tour_length(x, y_hat)
            opt_gap     = l - l_opt
            opt_gaps.append(opt_gap)
        opt_gap = torch.cat(opt_gaps).mean().item()
        loss = np.array(losses).mean().item()
        info = {'train_opt_gap': opt_gap, 'train_loss': loss}
        return info

    def test_epoch(self):
        # extract args
        dataloader = self.test_dataloader
        args       = self.args
        net        = self.net
        # evaluate each batch
        opt_gaps = []
        for i, batch in enumerate(dataloader):
            x, y     = batch.values()
            x = x.to(args.device)
            y = y.to(args.device)
            _, y_hat = net(x)
            l_opt    = util.get_tour_length(x, y)
            l        = util.get_tour_length(x, y_hat)
            opt_gap  = l - l_opt
            opt_gaps.append(opt_gap)
        opt_gap = torch.cat(opt_gaps).mean().item()
        info = {'test_opt_gap': opt_gap}
        return info

    def train(self):
        # extract args
        monitor = self.monitor
        args    = self.args
        self.best_opt_gap = np.inf
        #
        for epoch in range(args.n_epoch):
            info = {'epoch': epoch}
            _ = self.train_epoch()
            info.update(_)
            _ = self.test_epoch()
            info.update(_)
            monitor.step(info)
            monitor.export_csv()
            if info['train_opt_gap'] < self.best_opt_gap:
                self.save()
                self.best_opt_gap = info['train_opt_gap']

    def save(self):
        path = os.path.join(self.args.model_dir, f'{self.monitor.label}_{self.best_opt_gap:0.2f}.pkl')
        torch.save(self.net.state_dict(), path)
