import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle
import torch
import math
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
            yield l.detach().cpu().numpy()

    def test(self):
        # extract args
        args = self.args
        monitor = self.monitor
        monitor.create_progress_bar(math.ceil(args.n_test_instance / args.batch_size))
        args.n_logging = 1
        self.load_model()
        self.actor.eval()
        ls = []
        for l in self.test_epoch():
            ls.extend(l.tolist())
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
