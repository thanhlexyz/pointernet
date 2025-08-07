import simulator as sim
import torch

from .base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        action = self.env.y[self.env.n_step]
        return action
