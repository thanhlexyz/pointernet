import simulator as sim
import torch

from .base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        visited = observation[:, -1]
        indices = torch.where(visited == 0)[0]
        N = len(indices)
        probs = torch.full([N], 1. / float(N))
        choice = torch.multinomial(probs, 1)
        action = indices[choice]
        return action
