from .dataset import create_dataset
from .net import create_net

from .monitor import Monitor
from .solver import Solver

def create_solver(args):
    return Solver(args)
