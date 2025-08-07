import simulator as sim
import torch

from .base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        visited = observation[:, -1]
        indices = torch.where(visited == 0)[0]
        x = observation[:, :-1]
        x1 = x[self.env.action]
        d = ((x - x1[None, :])**2).sum(dim=1)[indices]
        action = indices[torch.argmin(d)]
        return action
