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
        self.critic_loss_fn = torch.nn.MSELoss()
        self.actor_optimizer = \
            optim.Adam(filter(lambda p: p.requires_grad, self.actor.parameters()),
                       lr=args.lr)
        self.actor_scheduler = \
            optim.lr_scheduler.StepLR(self.actor_optimizer,
                                      step_size=args.lrs_step_size,
                                      gamma=args.lrs_gamma)
        self.critic_optimizer = \
            optim.Adam(filter(lambda p: p.requires_grad, self.critic.parameters()),
                       lr=args.lr)
        self.critic_scheduler = \
            optim.lr_scheduler.StepLR(self.critic_optimizer,
                                      step_size=args.lrs_step_size,
                                      gamma=args.lrs_gamma)
        if args.load_state_dict:
            self.load_model()

    def train_epoch(self):
        # extract args
        args = self.args
        dataloader = self.dataloader_dict['train']
        # extract model
        actor, actor_optimizer, actor_scheduler = \
            self.actor, self.actor_optimizer, self.actor_scheduler
        critic, critic_loss_fn, critic_optimizer, critic_scheduler = \
            self.critic, self.critic_loss_fn, self.critic_optimizer, self.critic_scheduler
        # training loop
        for x, y in dataloader:
            # extract data
            x = x.to(args.device)
            y = y.to(args.device)
            # get actor prediction
            log_likelihood, y_hat = actor(x)
            # optimize critic
            l = util.get_tour_length(x, y_hat)
            l_hat = critic(x)
            critic_loss = critic_loss_fn(l_hat, l.detach())
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(),
                                     max_norm=1., norm_type=2)
            critic_optimizer.step()
            critic_scheduler.step()
            # optimize actor
            advantage = l.detach() - l_hat.detach()
            actor_loss = (advantage * log_likelihood).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(),
                                     max_norm=1., norm_type=2)
            actor_optimizer.step()
            actor_scheduler.step()
            # gather info
            info = {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'avg_tour_length': l.mean().item(),
            }
            yield info

    def train(self):
        # extract args
        args       = self.args
        dataloader = self.dataloader_dict['train']
        monitor    = self.monitor
        n_step     = len(dataloader) * args.n_train_epoch
        self.actor.train()
        self.critic.train()
        #
        monitor.create_progress_bar(n_step)
        self.step = 0
        try:
            for epoch in range(args.n_train_epoch):
                for info in self.train_epoch():
                    info.update({'step': self.step, 'epoch': epoch})
                    if self.step % args.n_logging == 0:
                        monitor.step(info)
                    self.step += 1
        except KeyboardInterrupt:
            pass
        finally:
            self.save_model()
            monitor.export_csv(mode='train')

    def test_epoch(self):
        # extract args
        args = self.args
        dataloader = self.dataloader_dict['test']
        # extract model
        actor = self.actor
        # training loop
        for x, y in dataloader:
            # extract data
            x = x.to(args.device)
            # get actor prediction
            _, y_hat = actor(x)
            l = util.get_tour_length(x, y_hat)
            # gather info
            yield l

    def test(self):
        # extract args
        monitor = self.monitor
        self.load_model()
        self.actor.eval()
        ls = []
        for l in self.test_epoch():
            ls.append(l)
        ls = torch.cat(ls).detach().cpu().numpy()
        info = {'avg_tour_length': np.mean(ls)}
        monitor.step(info)
        monitor.export_csv(mode='test')

    def save_model(self):
        args = self.args
        # save model
        path = os.path.join(args.model_dir, f'{self.label}.pkl')
        data = {'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict()}
        with open(path, 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'[+] saved at {self.step=} {path=}')

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
