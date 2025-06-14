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
        # create actor/critic model
        self.create_model()
        # load monitor
        self.monitor = lib.Monitor(args)

    @property
    def label(self):
        args = self.args
        return f'{args.dataset}_{args.n_node_min}_{args.n_node_max}'

    def create_model(self):
        args = self.args
        self.actor  = lib.pointer_net.Actor(args)
        self.critic = lib.pointer_net.Critic(args)
        # self.loss_function = torch.nn.CrossEntropyLoss()
        self.critic_loss_fn = torch.nn.MSELoss()
        self.actor_optimizer = \
            optim.Adam(filter(lambda p: p.requires_grad, self.actor.parameters()),
                       lr=args.lr)
        self.actor_scheduler = \
            optim.lr_scheduler.StepLR(self.actor_optimizer,
                                      step_size=args.lrs_step_size,
                                      gamma=args.lrs_gamma)
        self.critic_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.critic.parameters()),
                                          lr=args.lr)
        self.critic_scheduler = \
            optim.lr_scheduler.StepLR(self.critic_optimizer,
                                      step_size=args.lrs_step_size,
                                      gamma=args.lrs_gamma)

    def shuffle(self, x):
        args = self.args
        x_shuffle = torch.zeros_like(x)
        for i in range(args.batch_size):
            x_shuffle[i] = x[i, torch.randperm(args.n_node)]
        return x_shuffle

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
        x = torch.tensor(x, device=args.device)
        y = torch.tensor(y, device=args.device)
        # replicate x for batch processing:
        x = x.repeat(args.batch_size, 1, 1)
        # get baseline tour length from a random tour
        y_rand = torch.stack([torch.randperm(args.n_node) for i in range(args.batch_size)], dim=0)
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

    def test_epoch(self):
        # extract args
        args = self.args
        dataset = self.dataloader_dict['test'].dataset
        # sampling loop
        for item in tqdm.tqdm(dataset):
            yield self.test_item(item)

    def test(self):
        # extract args
        args = self.args
        monitor = self.monitor
        ls = []
        for l in self.test_epoch():
            ls.append(l)
        ls = np.array(ls)
        info = {'avg_tour_length': np.mean(ls)}
        monitor.step(info)
        monitor.export_csv(mode='test')

    def load_model(self):
        args = self.args
        path = os.path.join(args.model_dir, f'{self.label}.pkl')
        if os.path.exists(path):
            if args.verbose:
                print(f'[+] loading {path}')
            with open(path, 'rb') as fp:
                data = pickle.load(fp)
            self.actor.load_state_dict(data['actor'])
            self.critic.load_state_dict(data['critic'])
            self.actor.eval()
            self.critic.eval()
