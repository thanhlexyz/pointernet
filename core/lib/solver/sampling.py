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
        return f'{args.dataset}_{args.n_node}'

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

    def test_epoch(self):
        # extract args
        args = self.args
        dataset = self.dataloader_dict['test'].dataset
        # extract model
        actor = self.actor
        # sampling loop
        for item in tqdm.tqdm(dataset):
            x, y = item.values()
            l_best = torch.inf
            x = torch.tensor(x, device=args.device)
            y = torch.tensor(y, device=args.device)
            x = x.repeat(args.sampling_batch_size, 1, 1)
            # get actor prediction
            _, y_hat = actor(x)
            l = util.get_tour_length(x, y_hat)
            idx = torch.argmin(l)
            l = l[idx]
            y_hat = y_hat[idx]
            if l < l_best:
                l_best = l.item()
            # gather info
            yield l_best

    def test(self):
        # extract args
        args = self.args
        monitor = self.monitor
        self.load_model()
        self.actor.eval()
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
            print(f'[+] loading {path}')
            with open(path, 'rb') as fp:
                data = pickle.load(fp)
            self.actor.load_state_dict(data['actor'])
            self.critic.load_state_dict(data['critic'])
            self.actor.eval()
            self.critic.eval()
