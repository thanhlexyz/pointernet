import torch.optim as optim
import numpy as np
import pickle
import torch
import math
import os

import lib


class SolverBase:

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
            
    @property
    def label(self):
        args = self.args
        return f'{args.dataset}_{args.n_node_min}_{args.n_node_max}'

    def test_epoch(self):
        dataset = self.dataloader_dict['test'].dataset
        # sampling loop
        for item in dataset:
            yield self.test_item(item)

    def test(self):
        # extract args
        args = self.args
        monitor = self.monitor
        monitor.create_progress_bar(self.n_test)
        args.n_logging = 1
        ls = []
        for l in self.test_epoch():
            if isinstance(l, list):
                ls.extend(l)
            else:
                ls.append(l)
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
