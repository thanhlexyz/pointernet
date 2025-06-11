from .monitor import Monitor
from .solver import Solver
from . import pointer_net, dataset

def create_solver(args):
    return Solver(args)
